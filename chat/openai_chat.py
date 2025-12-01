import os
import time
from typing import List, Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from .base import BaseLLM, Message, ChatResponse


class OpenAIChat(BaseLLM):
    """OpenAI API implementation"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        **kwargs,
    ):
        """
        Initialize OpenAI Chat

        Args:
            model: Model name, default is gpt-4o-mini
            api_key: API key, if not provided will read from OPENAI_API_KEY env variable
            max_retries: Maximum number of retries for rate limit errors (default: 5)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            **kwargs: Other config parameters
        """
        super().__init__(model, **kwargs)

        # Configure API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide api_key or set OPENAI_API_KEY environment variable"
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
        """
        Generate response with automatic retry on rate limit errors

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other OpenAI-specific parameters

        Returns:
            ChatResponse: Generated response
        """
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # Update with other configs
        api_params.update(kwargs)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API
                response = self.client.chat.completions.create(**api_params)

                # Extract usage information
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

            except RateLimitError as e:
                last_exception = e
                wait_time = self.retry_delay * (2**attempt)  # Exponential backoff

                if attempt < self.max_retries - 1:
                    print(
                        f"[Rate limit hit] Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise last_exception

            except APIError as e:
                last_exception = e
                error_str = str(e)

                # Check if it's a retryable error
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = self.retry_delay * (2**attempt)

                    if attempt < self.max_retries - 1:
                        print(
                            f"[API Error - Rate limit] Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}..."
                        )
                        time.sleep(wait_time)
                        continue

                # For non-retryable errors, raise immediately
                raise last_exception

            except Exception as e:
                # For unexpected errors, raise immediately
                raise e

        # If all retries exhausted
        raise last_exception
