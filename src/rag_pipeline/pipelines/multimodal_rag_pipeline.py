"""Multimodal RAG pipeline combining vision-based and text-based retrieval.

Stage 1: ColPali vision-based page retrieval
Stage 2: Text-based chunk retrieval on retrieved pages
Stage 3: Optional reranking
Stage 4: LLM generation
"""

import os
import re
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


class MultimodalRAGPipeline(BaseRAGPipeline):
    """Multimodal RAG: ColPali page retrieval -> text chunk retrieval -> generation.

    Expects:
    - .pt embedding file in doc_dir
    - Markdown files named {pdf_name}_page_{X}.md in doc_dir
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
    ):
        """Initialize multimodal RAG pipeline.

        Args:
            doc_dir: Directory containing .pt, .pdf, and .md files
            vision_top_k: Number of pages to retrieve using ColPali
            text_top_k: Number of text chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            llm_model: LLM model name for generation
            embedding_model: Embedding model name for text retrieval
            use_reranker: Whether to use reranker
            reranker_model: Reranker model name
            rerank_top_k: Number of chunks after reranking (defaults to text_top_k)
        """
        self.doc_dir = doc_dir
        self.vision_top_k = vision_top_k
        self.text_top_k = text_top_k
        self.chunker = SimpleChunker(chunk_size, chunk_overlap)
        self.llm: BaseLLM = GeminiChat(model=llm_model)
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else text_top_k

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

        print(f"[INFO] Multimodal pipeline initialized")

    def _preprocess_text(self, text: str) -> str:
        """Clean OCR artifacts from text.

        Removes detection box annotations like:
        - <|ref|>title<|/ref|><|det|>[[141, 174, 263, 190]]<|/det|>
        - <|det|>[[x, y, w, h]]<|/det|>

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned text
        """
        # Remove detection box tags: <|det|>[[coordinates]]<|/det|>
        text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)

        # Remove reference tags but keep content: <|ref|>content<|/ref|> -> content
        text = re.sub(r'<\|ref\|>(.*?)<\|/ref\|>', r'\1', text)

        # Clean up multiple spaces and newlines
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _load_pages(self, page_nums: List[int]) -> List[Document]:
        """Load markdown content for specific pages.

        Args:
            page_nums: List of page numbers to load

        Returns:
            List of Document objects with chunked text
        """
        documents = []

        # Find PDF name from markdown files in directory
        import glob
        md_files = glob.glob(os.path.join(self.doc_dir, "*.md"))
        if not md_files:
            raise ValueError(f"No markdown files found in {self.doc_dir}")

        # Extract PDF name from first markdown file
        # Pattern: {pdf_name}_page_{X}.md
        pattern = re.compile(r'^(.+)_page_(\d+)\.md$')
        first_file = os.path.basename(md_files[0])
        match = pattern.match(first_file)
        if not match:
            raise ValueError(f"Cannot extract PDF name from {first_file}")

        pdf_name = match.group(1) + ".pdf"

        # Load each requested page
        for page_num in page_nums:
            md_filename = f"{match.group(1)}_page_{page_num}.md"
            md_path = os.path.join(self.doc_dir, md_filename)

            if not os.path.exists(md_path):
                print(f"Warning: {md_path} not found, skipping page {page_num}")
                continue

            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Warning: Failed to read {md_path}: {e}")
                continue

            if not text.strip():
                continue

            # Preprocess to remove OCR artifacts
            text = self._preprocess_text(text)

            # Chunk the page text
            chunks = self.chunker.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": pdf_name,
                    "page_num": page_num,
                    "chunk_idx": chunk_idx,
                    "source": md_path,
                }
                documents.append(Document(page_content=chunk, metadata=metadata))

        print(f"[INFO] Loaded {len(documents)} chunks from {len(page_nums)} pages")
        return documents

    def query(self, question: str) -> str:
        """Run multimodal RAG query.

        Args:
            question: User's question

        Returns:
            Generated answer
        """
        response = self.query_with_metadata(question)
        return response.answer

    def query_with_metadata(self, question: str) -> RAGResponse:
        """Run multimodal RAG query and return answer with metadata.

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

        # Stage 2: Load markdown for retrieved pages
        with latency_context("LoadPages"):
            page_documents = self._load_pages(retrieved_page_nums)

        if not page_documents:
            return RAGResponse(
                answer="No relevant content found.",
                metadata={
                    "vision_retrieved_pages": retrieved_page_nums,
                    "text_retrieved_pages": [],
                    "final_pages": [],
                }
            )

        # Stage 3: Text-based chunk retrieval on retrieved pages
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
