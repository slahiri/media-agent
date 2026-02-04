"""Ollama LLM provider for local models."""

import json
from typing import Any

from .base import BaseLLM, LLMResponse, ToolCall


class OllamaLLM(BaseLLM):
    """Ollama LLM provider for locally running models."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: str | None = None,  # Not used, for interface compatibility
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the Ollama LLM.

        Args:
            model: Model name (llama3.2, mistral, codellama, etc.).
            api_key: Not used (Ollama doesn't require auth).
            base_url: Ollama server URL (default: http://localhost:11434).
            **kwargs: Additional arguments.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )
        self._client = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "ollama"

    @property
    def client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            try:
                from ollama import Client
            except ImportError:
                raise ImportError(
                    "Ollama package not installed. Install with: pip install ollama"
                )

            self._client = Client(host=self.base_url)
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
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs,
        )

        return response["message"]["content"]

    def supports_tools(self) -> bool:
        """Ollama supports tool calling for some models."""
        # Tool calling support depends on the model
        # Models like llama3.1+ and mistral-nemo support it
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
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }

        # Convert OpenAI tool format to Ollama format
        ollama_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                ollama_tools.append(tool)
            else:
                # Wrap in function format
                ollama_tools.append({
                    "type": "function",
                    "function": tool,
                })

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=ollama_tools,
                options=options,
                **kwargs,
            )

            message = response["message"]
            content = message.get("content", "")

            tool_calls = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", {}),
                            id=tc.get("id", ""),
                        )
                    )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=response,
            )

        except Exception as e:
            # Fall back to regular chat if tools not supported
            if "does not support tools" in str(e).lower():
                content = self.chat(messages, max_tokens, temperature, **kwargs)
                return LLMResponse(content=content, tool_calls=[])
            raise

    def unload(self) -> None:
        """Unload client."""
        self._client = None

    def is_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None

    def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model names.
        """
        try:
            response = self.client.list()
            return [model["name"] for model in response.get("models", [])]
        except Exception:
            return []

    def pull_model(self, model: str) -> bool:
        """Pull a model from the Ollama library.

        Args:
            model: Model name to pull.

        Returns:
            True if successful.
        """
        try:
            self.client.pull(model)
            return True
        except Exception:
            return False
