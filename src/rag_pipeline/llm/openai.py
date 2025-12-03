"""OpenAI LLM implementation with automatic retry."""

import os
from typing import List, Optional
from openai import OpenAI

from .base import BaseLLM, Message, ChatResponse
from .retry import with_exponential_backoff, is_rate_limit_error


class OpenAIChat(BaseLLM):
    """OpenAI API implementation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        **kwargs,
    ):
        """Initialize OpenAI Chat.

        Args:
            model: OpenAI model name
            api_key: API key (defaults to OPENAI_API_KEY env var)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial retry delay in seconds
            **kwargs: Additional configuration
        """
        super().__init__(model, **kwargs)

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """Generate response with automatic retry on rate limits."""
        api_params = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # Apply retry decorator dynamically
        @with_exponential_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_delay,
            should_retry=is_rate_limit_error,
        )
        def _call_api():
            return self.client.chat.completions.create(**api_params)

        response = _call_api()

        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage=usage,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            },
        )
