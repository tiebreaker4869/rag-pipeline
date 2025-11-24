from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
from langchain_core.documents import Document
from collections import defaultdict
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def rrf_fuse(doc_lists, weights=None, c=60) -> List[Document]:
    if weights is None:
        weights = [1 / len(doc_lists)] * len(doc_lists)
    scores = defaultdict(float)
    metas = {}

    for w, docs in zip(weights, doc_lists):
        for r, d in enumerate(docs, start=1):
            key = d.page_content
            scores[key] += w / (c + r)
            metas[key] = d.metadata

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [type(docs[0])(page_content=k, metadata=metas[k]) for k, _ in fused]


class HybridRetriever:
    # Class-level cache for embedding models
    _embedding_cache = {}

    def __init__(
        self,
        keyword_k: int,
        dense_k: int,
        documents: List[Document],
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize Hybrid Retriever with BM25 and Dense retrieval

        Args:
            keyword_k: Number of documents to retrieve for BM25
            dense_k: Number of documents to retrieve for dense retrieval
            documents: List of documents to index
            embedding_model: Embedding model name. Supported:
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                - HuggingFace: "BAAI/bge-large-en-v1.5" (default), "BAAI/bge-small-en-v1.5",
                               "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"
                - Or any HuggingFace model name
        """
        # Create embeddings based on model name
        if embedding_model is None:
            embedding_model = "BAAI/bge-large-en-v1.5"

        # Parallelize BM25 and FAISS indexing
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both indexing tasks
            bm25_future = executor.submit(self._build_bm25_index, documents, keyword_k)
            dense_future = executor.submit(
                self._build_dense_index, documents, embedding_model, dense_k
            )

            # Wait for both to complete
            self.bm25 = bm25_future.result()
            self.dense = dense_future.result()

    def _build_bm25_index(self, documents: List[Document], keyword_k: int):
        """Build BM25 index"""
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = keyword_k
        return bm25

    def _build_dense_index(
        self, documents: List[Document], embedding_model: str, dense_k: int
    ):
        """Build FAISS dense index"""
        embeddings = self._get_or_create_embeddings(embedding_model)
        vs = FAISS.from_documents(documents, embeddings)
        return vs.as_retriever(search_kwargs={"k": dense_k})

    def _get_or_create_embeddings(self, model_name: str):
        """Get embeddings from cache or create new one"""
        if model_name in self._embedding_cache:
            print(f"[INFO] Reusing cached embedding model: {model_name}")
            return self._embedding_cache[model_name]

        print(f"[INFO] Loading embedding model: {model_name}")
        embeddings = self._create_embeddings(model_name)
        self._embedding_cache[model_name] = embeddings
        return embeddings

    def _create_embeddings(self, model_name: str):
        """Create embeddings from model name"""
        # Check if it's an OpenAI model
        openai_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        if model_name in openai_models:
            # Use OpenAI embeddings
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    f"OPENAI_API_KEY environment variable is required for model {model_name}"
                )
            return OpenAIEmbeddings(model=model_name)
        else:
            # Use HuggingFace embeddings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )

    def retrieve(
        self, question: str, top_k: int, weights: List[float] = None
    ) -> List[Document]:
        # Parallelize BM25 and dense retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25.invoke, question)
            dense_future = executor.submit(self.dense.invoke, question)

            bm25_docs = bm25_future.result()
            dense_docs = dense_future.result()

        results = rrf_fuse([bm25_docs, dense_docs], weights)
        return results[:top_k]
