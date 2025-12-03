from typing import List, Optional
from langchain_core.documents import Document
from .base import BaseReranker
import torch


class BGEReranker(BaseReranker):
    """BGE Cross-Encoder based reranker for precise document ranking"""

    # Class-level cache for reranker models
    _model_cache = {}

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
    ):
        """
        Initialize BGE Reranker

        Args:
            model_name: Model name from Hugging Face
                - "BAAI/bge-reranker-base" (default, ~278M params, good balance)
                - "BAAI/bge-reranker-large" (~560M params, best quality)
                - "BAAI/bge-reranker-v2-m3" (multilingual support)
            device: Device to run model on. If None, auto-detects cuda/mps/cpu
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Load model from cache or create new
        self.model = self._get_or_create_model(model_name, device)

    def _get_or_create_model(self, model_name: str, device: str):
        """Get model from cache or create new one"""
        cache_key = f"{model_name}_{device}"

        if cache_key in self._model_cache:
            print(f"[INFO] Reusing cached reranker model: {model_name}")
            return self._model_cache[cache_key]

        print(f"[INFO] Loading reranker model: {model_name} on {device}")
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device=device)
        self._model_cache[cache_key] = model
        return model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank documents using BGE cross-encoder

        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Number of top documents to return, None returns all

        Returns:
            List[Document]: Reranked documents sorted by relevance score
        """
        if not documents:
            return []

        # Prepare input pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Compute relevance scores
        scores = self.model.predict(pairs)

        # Convert numpy array to list if needed
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # Pair documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]

        return [doc for doc, _ in doc_scores]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents and return with scores

        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List[tuple[Document, float]]: List of (document, score) tuples
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            doc_scores = doc_scores[:top_k]

        return doc_scores
