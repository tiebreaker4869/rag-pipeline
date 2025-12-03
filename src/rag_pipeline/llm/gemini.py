"""Gemini LLM implementation with automatic retry."""

import os
from typing import List, Optional
import google.generativeai as genai

from .base import BaseLLM, Message, ChatResponse
from .retry import with_exponential_backoff, is_rate_limit_error


class GeminiChat(BaseLLM):
    """Google Gemini API implementation."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        **kwargs,
    ):
        """Initialize Gemini Chat.

        Args:
            model: Gemini model name
            api_key: API key (defaults to GEMINI_API_KEY env var)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial retry delay in seconds
            **kwargs: Additional configuration
        """
        super().__init__(model, **kwargs)

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY or pass api_key parameter."
            )

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """Generate response with automatic retry on rate limits."""
        generation_config = {"temperature": temperature, **kwargs}
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Apply retry decorator dynamically
        @with_exponential_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_delay,
            should_retry=is_rate_limit_error,
        )
        def _call_api():
            gemini_messages = self._convert_messages(messages)
            return self.client.generate_content(
                gemini_messages, generation_config=generation_config
            )

        response = _call_api()

        # Extract usage
        usage = None
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return ChatResponse(
            content=response.text,
            model=self.model,
            usage=usage,
            metadata={
                "finish_reason": (
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else None
                )
            },
        )

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """Convert generic messages to Gemini format.

        Gemini uses 'user' and 'model' roles. System prompts are
        prepended to the first user message.
        """
        gemini_messages = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                content = msg.content
                if system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                    system_prompt = None
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif msg.role == "assistant":
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": msg.content}]}
                )

        return gemini_messages
