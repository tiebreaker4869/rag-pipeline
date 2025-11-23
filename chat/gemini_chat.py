import os
from typing import List, Optional
import google.generativeai as genai
from .base import BaseLLM, Message, ChatResponse


class GeminiChat(BaseLLM):
    """Gemini API implementation"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemini Chat

        Args:
            model: Model name, default is gemini-2.0-flash-exp
            api_key: API key, if not provided will read from GEMINI_API_KEY env variable
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

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Generate response

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

        # Call Gemini API
        response = self.client.generate_content(
            gemini_messages,
            generation_config=generation_config
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
                "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None
            }
        )

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
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif msg.role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })

        return gemini_messages
