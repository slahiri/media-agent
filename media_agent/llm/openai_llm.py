"""OpenAI LLM provider."""

import json
from typing import Any

from .base import BaseLLM, LLMResponse, ToolCall


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider (GPT-4, GPT-3.5, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the OpenAI LLM.

        Args:
            model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.).
            api_key: OpenAI API key.
            base_url: Optional base URL override.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )

            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set it with /settings openai.api_key <key>")

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens, temperature, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response in chat format."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def supports_tools(self) -> bool:
        """OpenAI supports tool calling."""
        return True

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with tool calling support."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        message = response.choices[0].message
        content = message.content or ""

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        id=tc.id,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response,
        )

    def unload(self) -> None:
        """Unload client (close connections)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def is_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None
