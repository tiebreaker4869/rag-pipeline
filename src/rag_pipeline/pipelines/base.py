"""Base interface for RAG pipelines."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RAGResponse:
    """Response from a RAG pipeline query.

    Attributes:
        answer: Generated answer text
        metadata: Optional metadata about the retrieval and generation process
            - retrieved_docs: Number of documents retrieved
            - reranked_docs: Number of documents after reranking (if applicable)
            - sources: Source documents/pages used
            - latencies: Timing information for each stage
    """
    answer: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines.

    All RAG pipelines should implement this interface to ensure
    compatibility with evaluation scripts and consistent usage.
    """

    @abstractmethod
    def query(self, question: str, doc_id: Optional[str] = None) -> str:
        """Run RAG query and return answer.

        Args:
            question: User's question
            doc_id: Optional document ID to restrict search scope.
                   If None, searches across all documents.

        Returns:
            Generated answer as string
        """
        pass

    def query_with_metadata(
        self,
        question: str,
        doc_id: Optional[str] = None
    ) -> RAGResponse:
        """Run RAG query and return answer with metadata.

        Default implementation calls query() and wraps result in RAGResponse.
        Subclasses can override to provide richer metadata.

        Args:
            question: User's question
            doc_id: Optional document ID to restrict search scope

        Returns:
            RAGResponse with answer and metadata
        """
        answer = self.query(question, doc_id)
        return RAGResponse(answer=answer)
