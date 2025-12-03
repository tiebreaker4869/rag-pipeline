"""Vector-based dense retrieval using FAISS."""

from typing import List, Optional
import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorRetriever:
    """Dense vector retrieval using FAISS index."""

    # Class-level cache for embedding models
    _embedding_cache = {}

    def __init__(
        self,
        documents: List[Document],
        embedding_model: Optional[str] = None,
        top_k: int = 5,
    ):
        """Initialize vector retriever with FAISS index.

        Args:
            documents: List of documents to index
            embedding_model: Embedding model name. Supported:
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                - HuggingFace: "BAAI/bge-large-en-v1.5" (default), "BAAI/bge-small-en-v1.5",
                               "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        self.embedding_model_name = embedding_model or "BAAI/bge-large-en-v1.5"

        # Get or create embeddings
        embeddings = self._get_or_create_embeddings(self.embedding_model_name)

        # Build FAISS index
        print(f"[INFO] Building FAISS index with {len(documents)} documents...")
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        print(f"[INFO] FAISS index built successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve top-k most relevant documents.

        Args:
            query: Search query
            top_k: Number of documents to retrieve (defaults to constructor value)

        Returns:
            List of retrieved documents, sorted by relevance
        """
        k = top_k or self.top_k

        # Update k if different from default
        if k != self.top_k:
            self.retriever.search_kwargs["k"] = k

        return self.retriever.invoke(query)

    def _get_or_create_embeddings(self, model_name: str):
        """Get embeddings from cache or create new instance."""
        if model_name in self._embedding_cache:
            print(f"[INFO] Reusing cached embedding model: {model_name}")
            return self._embedding_cache[model_name]

        print(f"[INFO] Loading embedding model: {model_name}")
        embeddings = self._create_embeddings(model_name)
        self._embedding_cache[model_name] = embeddings
        return embeddings

    def _create_embeddings(self, model_name: str):
        """Create embeddings instance from model name."""
        # OpenAI models
        openai_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        if model_name in openai_models:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    f"OPENAI_API_KEY environment variable required for {model_name}"
                )
            return OpenAIEmbeddings(model=model_name)
        else:
            # HuggingFace models
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
