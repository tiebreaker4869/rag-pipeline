from typing import List, Optional
from langchain_core.documents import Document
from chat.base import BaseLLM
from .base import BaseReranker
import json


class LLMReranker(BaseReranker):
    """LLM-based document reranker that filters irrelevant documents"""

    def __init__(
        self,
        llm: BaseLLM,
        temperature: float = 0.0,
        relevance_threshold: float = 0.5,
    ):
        """
        Initialize LLM Reranker

        Args:
            llm: LLM instance implementing BaseLLM interface
            temperature: Generation temperature, default 0.0 for deterministic results
            relevance_threshold: Relevance threshold (0-1), docs below this are filtered
        """
        self.llm = llm
        self.temperature = temperature
        self.relevance_threshold = relevance_threshold

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank documents using LLM and filter irrelevant ones

        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Number of top documents to return, None returns all relevant docs

        Returns:
            List[Document]: Reranked and filtered relevant documents
        """
        if not documents:
            return []

        # Build scoring prompt
        prompt = self._build_scoring_prompt(query, documents)

        # Call LLM for scoring
        response = self.llm.chat(
            user_message=prompt,
            temperature=self.temperature,
            max_tokens=4096,
        )

        # Parse scoring results
        scores = self._parse_scores(response, len(documents))

        # Sort by score and filter
        doc_scores = list(zip(documents, scores))
        # Filter docs below threshold
        relevant_docs = [
            (doc, score)
            for doc, score in doc_scores
            if score >= self.relevance_threshold
        ]
        # Sort by score descending
        relevant_docs.sort(key=lambda x: x[1], reverse=True)

        # Take top k
        if top_k is not None:
            relevant_docs = relevant_docs[:top_k]

        return [doc for doc, _ in relevant_docs]

    def _build_scoring_prompt(self, query: str, documents: List[Document]) -> str:
        """Build prompt for scoring"""
        doc_texts = []
        for i, doc in enumerate(documents):
            doc_text = f"Document {i}:\n{doc.page_content}\n"
            doc_texts.append(doc_text)

        prompt = f"""You are a professional document relevance evaluator. Please evaluate the relevance of each document to the given query.

Query: {query}

Documents:
{''.join(doc_texts)}

Evaluate the relevance of each document to the query and assign a score between 0 and 1:
- 1.0: Highly relevant, directly answers the query
- 0.7-0.9: Very relevant, contains useful information
- 0.4-0.6: Moderately relevant, some related content
- 0.1-0.3: Weakly relevant, minimal related information
- 0.0: Not relevant at all

Provide your evaluation in the following JSON format only, without any additional text:
{{
    "scores": [
        {{"doc_id": 0, "score": 0.0, "reason": "brief explanation"}},
        {{"doc_id": 1, "score": 0.0, "reason": "brief explanation"}},
        ...
    ]
}}"""

        return prompt

    def _parse_scores(self, response: str, num_docs: int) -> List[float]:
        """
        Parse LLM scoring response

        Args:
            response: LLM response
            num_docs: Number of documents

        Returns:
            List[float]: List of scores
        """
        try:
            # Extract JSON
            response = response.strip()

            # Handle code block markers
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            # Parse JSON
            result = json.loads(response)
            scores_data = result.get("scores", [])

            # Create score list
            scores = [0.0] * num_docs
            for item in scores_data:
                doc_id = item.get("doc_id")
                score = item.get("score", 0.0)
                if doc_id is not None and 0 <= doc_id < num_docs:
                    # Ensure score is in [0, 1]
                    scores[doc_id] = max(0.0, min(1.0, float(score)))

            return scores

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # If parsing fails, return default scores
            print(f"[WARNING] Failed to parse LLM scores: {e}")
            print(f"[WARNING] Response was: {response[:200]}...")
            # Return uniform scores to maintain original order
            return [0.5] * num_docs