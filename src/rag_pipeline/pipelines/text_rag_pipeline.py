"""Text-based RAG pipeline using vector search on OCR markdown files."""

import glob
import os
from typing import List, Optional
import re

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from rag_pipeline.llm import BaseLLM, GeminiChat
from rag_pipeline.chunking import SimpleChunker
from rag_pipeline.prompts import rag_generation_prompt
from rag_pipeline.rerankers import BaseReranker, BGEReranker
from rag_pipeline.utils.profile import latency_context
from .base import BaseRAGPipeline


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


class TextRAGPipeline(BaseRAGPipeline):
    """Text-only RAG: load OCR markdown -> chunk -> embed -> FAISS -> generation.

    Expects markdown files in doc_dir named as: {pdf_name}_page_{X}.md
    Indexes all pages from a single document.
    """

    def __init__(
        self,
        doc_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        top_k: int = 5,
        llm_model: str = "gemini-2.5-flash",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        rerank_top_k: Optional[int] = None,
    ):
        """Initialize text RAG pipeline.

        Args:
            doc_dir: Directory containing markdown files ({pdf_name}_page_{X}.md)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            llm_model: LLM model name for generation
            embedding_model: Embedding model name
            use_reranker: Whether to use reranker
            reranker_model: Reranker model name
            rerank_top_k: Number of chunks after reranking (defaults to top_k)
        """
        self.doc_dir = doc_dir
        self.chunker = SimpleChunker(chunk_size, chunk_overlap)
        self.llm: BaseLLM = GeminiChat(model=llm_model)
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else top_k

        # Initialize reranker if enabled
        self.reranker: Optional[BaseReranker] = None
        if use_reranker:
            self.reranker = BGEReranker(model_name=reranker_model)

        # Build index
        with latency_context("BuildIndex"):
            self.retriever, self.vectorstore = self._build_vector_index()

    def _build_vector_index(self):
        """Build FAISS vector index from markdown documents."""
        with latency_context("LoadAndChunk"):
            documents = self._load_documents()

        print(f"[INFO] Building FAISS index with {len(documents)} chunks...")
        embeddings = _create_embeddings(self.embedding_model)
        vs = FAISS.from_documents(documents, embeddings)
        retriever = vs.as_retriever(search_kwargs={"k": self.top_k})
        return retriever, vs

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

    def _load_documents(self) -> List[Document]:
        """Load documents from OCR markdown files.

        Expects files named: {pdf_name}_page_{X}.md
        """
        md_paths = glob.glob(os.path.join(self.doc_dir, "*.md"))
        if not md_paths:
            raise ValueError(f"No markdown files found in {self.doc_dir}")

        print(f"[INFO] Found {len(md_paths)} markdown files")
        docs: List[Document] = []

        # Pattern to extract PDF name and page number from filename
        # e.g., "document_name_page_5.md" -> pdf_name="document_name.pdf", page_num=5
        pattern = re.compile(r'^(.+)_page_(\d+)\.md$')

        for md_path in md_paths:
            filename = os.path.basename(md_path)
            match = pattern.match(filename)

            if not match:
                print(f"Warning: Skipping {filename} (doesn't match naming pattern)")
                continue

            pdf_name = match.group(1) + ".pdf"
            page_num = int(match.group(2))

            # Read markdown content
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
                docs.append(Document(page_content=chunk, metadata=metadata))

        print(f"[INFO] Loaded {len(docs)} chunks from {len(md_paths)} pages")
        return docs

    def query(self, question: str, doc_id: Optional[str] = None) -> str:
        """Run RAG query.

        Args:
            question: User's question
            doc_id: Ignored (kept for interface compatibility)

        Returns:
            Generated answer
        """
        # Retrieve
        with latency_context("TextRetrieval"):
            retrieved_documents = self.retriever.invoke(question)

        # Rerank if enabled
        if self.use_reranker and self.reranker:
            with latency_context("Rerank"):
                retrieved_documents = self.reranker.rerank(
                    question, retrieved_documents, top_k=self.rerank_top_k
                )

        # Generate
        with latency_context("FinalGeneration"):
            context = self._create_context(retrieved_documents)
            page_numbers = self._format_page_numbers(retrieved_documents)
            prompt = rag_generation_prompt.format(
                question=question,
                page_numbers=page_numbers,
                context=context
            )
            answer = self.llm.chat(prompt)

        return answer

    def _create_context(self, documents: List[Document]) -> str:
        """Concatenate document contents into context string."""
        if documents:
            return "\n\n".join([d.page_content for d in documents])
        return "\n"

    def _format_page_numbers(self, documents: List[Document]) -> str:
        """Extract and format page numbers from retrieved documents."""
        if not documents:
            return "No pages retrieved"

        page_nums = set()
        for doc in documents:
            if "page_num" in doc.metadata:
                page_nums.add(doc.metadata["page_num"])

        if page_nums:
            sorted_pages = sorted(page_nums)
            return f"Pages {', '.join(map(str, sorted_pages))}"
        return "Page numbers not available"