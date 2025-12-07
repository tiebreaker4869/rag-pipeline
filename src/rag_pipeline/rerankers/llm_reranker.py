"""LLM-based reranker that filters chunks by relevance without requiring k."""

import re
from typing import List, Optional
from langchain_core.documents import Document

from .base import BaseReranker
from ..llm.base import BaseLLM, Message


class LLMReranker(BaseReranker):
    """LLM-based reranker that identifies and returns all relevant chunks."""

    def __init__(
        self,
        llm: BaseLLM,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 4000,
    ):
        """
        Initialize LLM Reranker.

        Args:
            llm: LLM instance to use for reranking
            temperature: Temperature for LLM generation (default: 0.0 for deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, query: str, documents: List[Document]) -> str:
        """Build the reranking prompt with query and document chunks."""
        # Format chunks with indices
        chunks_text = ""
        for idx, doc in enumerate(documents):
            chunks_text += f"<chunk index=\"{idx}\">\n{doc.page_content}\n</chunk>\n\n"

        prompt = f"""You are a document understanding agent tasked with identifying the most relevant chunk(s) for a given user query. You will be presented with multiple text chunks and a user query. Your task is to determine which chunk(s) are relevant to answering the query.

First, review the text chunks:
<chunks>
{chunks_text.strip()}
</chunks>

Now, consider the following user query:
<user_query>
{query}
</user_query>

Important context about your task:
1. You are identifying chunks that contain information relevant to the user query.
2. A chunk is relevant if it contains information that helps answer the query, even partially.
3. It's better to include a potentially relevant chunk than to exclude it.
4. Multiple chunks may be relevant - include all that seem helpful.
5. If a chunk seems even somewhat related or contains terminology connected to the query, include it.
6. The chunk order should be from most relevant to least relevant in your answer.
7. Only exclude chunks that are clearly irrelevant or off-topic.

To determine which chunks are relevant:
1. Identify keywords, topics, and themes in the query.
2. Select any chunk(s) that contain information related to the query.
3. Be inclusive rather than exclusive - relevance can be partial or indirect.
4. Consider both direct answers and contextual information that supports answering the query.

After your analysis, provide your final answer in the following format:
<reasoning>
[Brief explanation of your relevance assessment...]
</reasoning>
<selected_chunks>
[List the indices of selected chunks, separated by commas. For example: 0,2,5,7]
</selected_chunks>

Remember: Only output the indices as numbers separated by commas, nothing else in the selected_chunks tags."""

        return prompt

    def _parse_selected_indices(self, response: str) -> List[int]:
        """Parse selected chunk indices from LLM response."""
        # Extract content between <selected_chunks> tags
        match = re.search(
            r"<selected_chunks>\s*(.*?)\s*</selected_chunks>",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if not match:
            # Fallback: try to find any comma-separated numbers
            numbers = re.findall(r"\b\d+\b", response)
            if numbers:
                return [int(n) for n in numbers]
            return []

        indices_text = match.group(1).strip()

        if not indices_text:
            return []

        # Parse comma-separated indices
        try:
            indices = [int(idx.strip()) for idx in indices_text.split(",")]
            return indices
        except ValueError:
            # If parsing fails, try to extract any numbers from the text
            numbers = re.findall(r"\b\d+\b", indices_text)
            return [int(n) for n in numbers]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank documents using LLM to identify all relevant chunks.

        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return.
                   If None, returns all relevant documents as determined by LLM.

        Returns:
            List[Document]: Relevant documents ordered by relevance
        """
        if not documents:
            return []

        # Build prompt
        prompt = self._build_prompt(query, documents)

        # Call LLM
        messages = [Message(role="user", content=prompt)]
        response = self.llm.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse selected indices
        selected_indices = self._parse_selected_indices(response.content)

        # Filter invalid indices
        valid_indices = [
            idx for idx in selected_indices if 0 <= idx < len(documents)
        ]

        # Get selected documents in order
        reranked_docs = [documents[idx] for idx in valid_indices]

        # Apply top_k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]

        return reranked_docs

    def rerank_with_reasoning(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> tuple[List[Document], str]:
        """
        Rerank documents and return both results and LLM reasoning.

        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return

        Returns:
            Tuple of (reranked documents, reasoning text)
        """
        if not documents:
            return [], ""

        # Build prompt
        prompt = self._build_prompt(query, documents)

        # Call LLM
        messages = [Message(role="user", content=prompt)]
        response = self.llm.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse reasoning
        reasoning_match = re.search(
            r"<reasoning>\s*(.*?)\s*</reasoning>",
            response.content,
            re.DOTALL | re.IGNORECASE,
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Parse selected indices
        selected_indices = self._parse_selected_indices(response.content)

        # Filter invalid indices
        valid_indices = [
            idx for idx in selected_indices if 0 <= idx < len(documents)
        ]

        # Get selected documents in order
        reranked_docs = [documents[idx] for idx in valid_indices]

        # Apply top_k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]

        return reranked_docs, reasoning
