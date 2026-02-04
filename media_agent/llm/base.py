"""Base LLM interface for multi-provider support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class LLMResponse:
    """Response from an LLM invocation."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the LLM.

        Args:
            model: Model name/identifier.
            api_key: API key for the provider.
            base_url: Optional base URL override.
            **kwargs: Provider-specific arguments.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Generated text string.
        """
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response in chat format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Assistant's response text.
        """
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if this LLM supports tool/function calling.

        Returns:
            True if tool calling is supported.
        """
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with tool calling support.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: List of tool definitions in OpenAI format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific arguments.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        ...

    def unload(self) -> None:
        """Unload model from memory. Override in subclasses if needed."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is currently loaded. Override in subclasses if needed."""
        return True
