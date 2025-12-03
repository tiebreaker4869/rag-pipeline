from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document


class BaseReranker(ABC):
    """Base interface for document rerankers"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank documents based on query relevance

        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Number of top documents to return, None returns all relevant docs

        Returns:
            List[Document]: Reranked and filtered documents
        """
        pass
