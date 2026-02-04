"""Anthropic LLM provider."""

import json
from typing import Any

from .base import BaseLLM, LLMResponse, ToolCall


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider (Claude models)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the Anthropic LLM.

        Args:
            model: Model name (claude-sonnet-4-20250514, claude-3-haiku, etc.).
            api_key: Anthropic API key.
            base_url: Optional base URL override.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    @property
    def client(self):
        """Lazy load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )

            if not self.api_key:
                raise ValueError("Anthropic API key is required. Set it with /settings anthropic.api_key <key>")

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = Anthropic(**kwargs)
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
        # Extract system message if present
        system = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        create_kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        if system:
            create_kwargs["system"] = system

        response = self.client.messages.create(**create_kwargs)

        # Extract text content
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "".join(text_parts)

    def supports_tools(self) -> bool:
        """Anthropic supports tool calling."""
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
        # Extract system message if present
        system = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)

        create_kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "tools": anthropic_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        if system:
            create_kwargs["system"] = system

        response = self.client.messages.create(**create_kwargs)

        # Extract content and tool calls
        text_parts = []
        tool_calls = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                        id=block.id,
                    )
                )

        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw_response=response,
        )

    def unload(self) -> None:
        """Unload client."""
        self._client = None

    def is_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None
