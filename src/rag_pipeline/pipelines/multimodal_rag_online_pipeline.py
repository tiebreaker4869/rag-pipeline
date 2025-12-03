"""Multimodal RAG pipeline with online PDF parsing.

Stage 1: ColPali vision-based page retrieval
Stage 2: Online parsing of retrieved pages
Stage 3: Text-based chunk retrieval on parsed pages
Stage 4: Optional reranking
Stage 5: LLM generation
"""

import os
from typing import List, Optional

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from rag_pipeline.llm import BaseLLM, GeminiChat
from rag_pipeline.retrievers import ColPaliRetriever
from rag_pipeline.chunking import SimpleChunker
from rag_pipeline.prompts import rag_generation_prompt
from rag_pipeline.rerankers import BaseReranker, BGEReranker
from rag_pipeline.parsers import PDFParser, PyMuPDFParser, TesseractPdfParser
from rag_pipeline.utils.profile import latency_context
from .base import BaseRAGPipeline, RAGResponse


# Global cache for embedding models
_EMBEDDING_CACHE = {}


def _create_embeddings(model_name: str):
    """Factory for embeddings with caching."""
    if model_name in _EMBEDDING_CACHE:
        print(f"[INFO] Reusing cached embedding model: {model_name}")
        return _EMBEDDING_CACHE[model_name]

    openai_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]

    if model_name in openai_models:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(f"OPENAI_API_KEY required for {model_name}")
        embeddings = OpenAIEmbeddings(model=model_name)
    else:
        print(f"[INFO] Loading embedding model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    _EMBEDDING_CACHE[model_name] = embeddings
    return embeddings


class MultimodalRAGOnlinePipeline(BaseRAGPipeline):
    """Multimodal RAG with online parsing: ColPali retrieval -> online parse -> text retrieval -> generation.

    Expects:
    - .pt embedding file in doc_dir
    - .pdf file in doc_dir
    """

    def __init__(
        self,
        doc_dir: str,
        vision_top_k: int = 10,
        text_top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        llm_model: str = "gemini-2.5-flash",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        rerank_top_k: Optional[int] = None,
        parser: Optional[PDFParser] = None,
        fallback_to_ocr: bool = True,
    ):
        """Initialize multimodal RAG pipeline with online parsing.

        Args:
            doc_dir: Directory containing .pt and .pdf files
            vision_top_k: Number of pages to retrieve using ColPali
            text_top_k: Number of text chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            llm_model: LLM model name for generation
            embedding_model: Embedding model name for text retrieval
            use_reranker: Whether to use reranker
            reranker_model: Reranker model name
            rerank_top_k: Number of chunks after reranking (defaults to text_top_k)
            parser: PDF parser instance (defaults to PyMuPDFParser with OCR fallback)
            fallback_to_ocr: Whether to fallback to OCR for scanned PDFs
        """
        self.doc_dir = doc_dir
        self.vision_top_k = vision_top_k
        self.text_top_k = text_top_k
        self.chunker = SimpleChunker(chunk_size, chunk_overlap)
        self.llm: BaseLLM = GeminiChat(model=llm_model)
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else text_top_k
        self.fallback_to_ocr = fallback_to_ocr

        # Find PDF file
        import glob
        pdf_files = glob.glob(os.path.join(doc_dir, "*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF file found in {doc_dir}")
        if len(pdf_files) > 1:
            print(f"Warning: Multiple PDF files found, using the first one: {pdf_files[0]}")

        self.pdf_path = pdf_files[0]
        print(f"[INFO] Using PDF: {self.pdf_path}")

        # Initialize parsers
        if parser is None:
            self.parser = PyMuPDFParser()
        else:
            self.parser = parser

        # Initialize OCR parser as fallback
        if fallback_to_ocr:
            self.ocr_parser = TesseractPdfParser()
        else:
            self.ocr_parser = None

        # Initialize ColPali retriever
        print(f"[INFO] Initializing ColPali retriever...")
        self.colpali_retriever = ColPaliRetriever(doc_dir=doc_dir, top_k=vision_top_k)

        # Initialize embedding model
        print(f"[INFO] Loading embedding model...")
        self.embeddings = _create_embeddings(embedding_model)

        # Initialize reranker if enabled
        self.reranker: Optional[BaseReranker] = None
        if use_reranker:
            self.reranker = BGEReranker(model_name=reranker_model)

        print(f"[INFO] Multimodal pipeline with online parsing initialized")

    def _parse_pages(self, page_nums: List[int]) -> List[Document]:
        """Parse specific pages from PDF.

        Args:
            page_nums: List of page numbers to parse

        Returns:
            List of Document objects with chunked text
        """
        documents = []

        # Parse requested pages using batch_process
        with latency_context("PDFParsing"):
            try:
                page_contents = self.parser.batch_process(
                    self.pdf_path,
                    page_nums=page_nums
                )

                # Check if we got empty text (possible scanned PDF)
                if self.fallback_to_ocr and self.ocr_parser:
                    empty_pages = [pc.page_num for pc in page_contents if not pc.text.strip()]
                    if empty_pages:
                        print(f"[INFO] Detected {len(empty_pages)} empty pages, using OCR fallback")
                        ocr_contents = self.ocr_parser.batch_process(
                            self.pdf_path,
                            page_nums=empty_pages
                        )
                        # Replace empty page contents with OCR results
                        ocr_dict = {pc.page_num: pc for pc in ocr_contents}
                        page_contents = [
                            ocr_dict.get(pc.page_num, pc) if not pc.text.strip() else pc
                            for pc in page_contents
                        ]

            except Exception as e:
                print(f"[ERROR] Primary parser failed: {e}")
                if self.fallback_to_ocr and self.ocr_parser:
                    print(f"[INFO] Falling back to OCR parser")
                    page_contents = self.ocr_parser.batch_process(
                        self.pdf_path,
                        page_nums=page_nums
                    )
                else:
                    raise

        # Extract PDF name for metadata
        pdf_name = os.path.basename(self.pdf_path)

        # Chunk each page
        for page_content in page_contents:
            page_num = page_content.page_num
            text = page_content.text

            if not text.strip():
                continue

            # Chunk the page text
            chunks = self.chunker.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": pdf_name,
                    "page_num": page_num,
                    "chunk_idx": chunk_idx,
                    "source": self.pdf_path,
                }
                documents.append(Document(page_content=chunk, metadata=metadata))

        print(f"[INFO] Parsed and chunked {len(documents)} chunks from {len(page_nums)} pages")
        return documents

    def query(self, question: str) -> str:
        """Run multimodal RAG query with online parsing.

        Args:
            question: User's question

        Returns:
            Generated answer
        """
        response = self.query_with_metadata(question)
        return response.answer

    def query_with_metadata(self, question: str) -> RAGResponse:
        """Run multimodal RAG query with online parsing and return answer with metadata.

        Args:
            question: User's question

        Returns:
            RAGResponse with answer and retrieval metadata
        """
        # Stage 1: ColPali vision-based page retrieval
        with latency_context("VisionRetrieval"):
            page_results = self.colpali_retriever.retrieve(question, top_k=self.vision_top_k)
            retrieved_page_nums = [result["page_num"] for result in page_results]

        print(f"[INFO] Retrieved pages: {retrieved_page_nums}")

        # Stage 2: Online parse retrieved pages
        page_documents = self._parse_pages(retrieved_page_nums)

        if not page_documents:
            return RAGResponse(
                answer="No relevant content found.",
                metadata={
                    "vision_retrieved_pages": retrieved_page_nums,
                    "text_retrieved_pages": [],
                    "final_pages": [],
                }
            )

        # Stage 3: Text-based chunk retrieval on parsed pages
        with latency_context("TextRetrieval"):
            vectorstore = FAISS.from_documents(page_documents, self.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": self.text_top_k})
            retrieved_documents = retriever.invoke(question)

        # Extract pages from text retrieval (before reranking)
        text_retrieved_pages = self._extract_page_numbers(retrieved_documents)

        # Stage 4: Rerank if enabled
        if self.use_reranker and self.reranker:
            with latency_context("Rerank"):
                retrieved_documents = self.reranker.rerank(
                    question, retrieved_documents, top_k=self.rerank_top_k
                )

        # Extract final pages (after reranking or text retrieval)
        final_pages = self._extract_page_numbers(retrieved_documents)

        # Stage 5: Generate answer
        with latency_context("FinalGeneration"):
            context = self._create_context(retrieved_documents)
            page_numbers = self._format_page_numbers(retrieved_documents)
            prompt = rag_generation_prompt.format(
                question=question,
                page_numbers=page_numbers,
                context=context
            )
            answer = self.llm.chat(prompt)

        # Cleanup
        del vectorstore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return RAGResponse(
            answer=answer,
            metadata={
                "vision_retrieved_pages": retrieved_page_nums,
                "text_retrieved_pages": text_retrieved_pages,
                "final_pages": final_pages,
                "num_chunks": len(retrieved_documents),
            }
        )

    def _create_context(self, documents: List[Document]) -> str:
        """Concatenate document contents into context string."""
        if documents:
            return "\n\n".join([d.page_content for d in documents])
        return "\n"

    def _extract_page_numbers(self, documents: List[Document]) -> List[int]:
        """Extract unique page numbers from documents.

        Args:
            documents: List of documents with page_num metadata

        Returns:
            Sorted list of unique page numbers
        """
        page_nums = set()
        for doc in documents:
            if "page_num" in doc.metadata:
                page_nums.add(doc.metadata["page_num"])
        return sorted(page_nums)

    def _format_page_numbers(self, documents: List[Document]) -> str:
        """Extract and format page numbers from retrieved documents."""
        if not documents:
            return "No pages retrieved"

        page_nums = self._extract_page_numbers(documents)

        if page_nums:
            return f"Pages {', '.join(map(str, page_nums))}"
        return "Page numbers not available"
