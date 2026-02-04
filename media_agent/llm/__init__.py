"""LLM module with multi-provider support."""

from media_agent.llm.base import BaseLLM, LLMResponse, ToolCall
from media_agent.llm.registry import (
    get_llm,
    get_provider_class,
    list_providers,
    register_provider,
    set_provider,
)
from media_agent.llm.settings import Settings, get_settings

# Keep QwenLLM for backwards compatibility
from media_agent.llm.qwen import QwenLLM

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMResponse",
    "ToolCall",
    # Registry
    "get_llm",
    "get_provider_class",
    "list_providers",
    "register_provider",
    "set_provider",
    # Settings
    "Settings",
    "get_settings",
    # Legacy
    "QwenLLM",
]
