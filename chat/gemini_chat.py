import os
import time
from typing import List, Optional
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from .base import BaseLLM, Message, ChatResponse


class GeminiChat(BaseLLM):
    """Gemini API implementation"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Gemini Chat

        Args:
            model: Model name, default is gemini-2.0-flash-exp
            api_key: API key, if not provided will read from GEMINI_API_KEY env variable
            max_retries: Maximum number of retries for rate limit errors (default: 5)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            **kwargs: Other config parameters
        """
        super().__init__(model, **kwargs)

        # Configure API key
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please provide api_key or set GEMINI_API_KEY environment variable"
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
        """
        Generate response with automatic retry on rate limit errors

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other Gemini-specific parameters

        Returns:
            ChatResponse: Generated response
        """
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)

        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Update with other configs
        generation_config.update(kwargs)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Call Gemini API
                response = self.client.generate_content(
                    gemini_messages, generation_config=generation_config
                )

                # Extract usage information
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

            except Exception as e:
                last_exception = e
                error_str = str(e)

                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    # Extract wait time from error message if available
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff

                    # Try to parse retry_delay from error message
                    if "retry in" in error_str.lower():
                        try:
                            import re
                            match = re.search(r'retry in ([\d.]+)s', error_str)
                            if match:
                                wait_time = float(match.group(1)) + 1  # Add 1s buffer
                        except:
                            pass

                    if attempt < self.max_retries - 1:
                        print(f"[Rate limit hit] Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                        time.sleep(wait_time)
                        continue

                # For non-rate-limit errors or last attempt, raise immediately
                raise last_exception

        # If all retries exhausted
        raise last_exception

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """
        Convert generic message format to Gemini format

        Gemini message format:
        - role: "user" or "model" (corresponds to assistant)
        - parts: [{"text": content}]
        """
        gemini_messages = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't support system role, use it as prefix of first user message
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
