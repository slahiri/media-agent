"""LLM provider registry and factory."""

from typing import Any, Type

from .base import BaseLLM
from .settings import get_settings


# Provider registry - lazy imports to avoid loading unused dependencies
PROVIDERS: dict[str, str] = {
    "openai": "media_agent.llm.openai_llm.OpenAILLM",
    "anthropic": "media_agent.llm.anthropic_llm.AnthropicLLM",
    "ollama": "media_agent.llm.ollama_llm.OllamaLLM",
    "huggingface": "media_agent.llm.huggingface_llm.HuggingFaceLLM",
}

# Default models for each provider
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.2",
    "huggingface": "Qwen/Qwen2.5-7B-Instruct",
}


def _import_class(class_path: str) -> Type[BaseLLM]:
    """Dynamically import a class from a module path.

    Args:
        class_path: Full path like 'media_agent.llm.openai_llm.OpenAILLM'.

    Returns:
        The class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_provider_class(provider: str) -> Type[BaseLLM]:
    """Get the LLM class for a provider.

    Args:
        provider: Provider name (openai, anthropic, ollama, huggingface).

    Returns:
        The LLM class.

    Raises:
        ValueError: If provider is not registered.
    """
    if provider not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")

    return _import_class(PROVIDERS[provider])


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseLLM:
    """Get an LLM instance for the specified provider.

    Args:
        provider: Provider name. Uses settings if not provided.
        model: Model name. Uses provider default if not provided.
        **kwargs: Additional arguments passed to the LLM constructor.

    Returns:
        Configured LLM instance.
    """
    settings = get_settings()

    # Use current provider from settings if not specified
    if provider is None:
        provider = settings.get_current_provider()

    # Get the LLM class
    llm_class = get_provider_class(provider)

    # Determine model to use
    if model is None:
        # Check settings for current model or provider default
        model = settings.get_current_model()
        if model is None:
            model = settings.get_default_model(provider)
        if model is None:
            model = DEFAULT_MODELS.get(provider, "")

    # Get API key and base URL from settings
    api_key = kwargs.pop("api_key", None) or settings.get_api_key(provider)
    base_url = kwargs.pop("base_url", None) or settings.get_base_url(provider)

    return llm_class(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def list_providers() -> list[dict[str, Any]]:
    """List all available providers with their status.

    Returns:
        List of provider info dicts.
    """
    settings = get_settings()
    current_provider = settings.get_current_provider()

    providers = []
    for name in PROVIDERS:
        has_key = settings.get_api_key(name) is not None
        default_model = settings.get_default_model(name) or DEFAULT_MODELS.get(name, "")

        # Check if provider requires API key
        requires_key = name in ("openai", "anthropic")

        providers.append({
            "name": name,
            "is_current": name == current_provider,
            "has_api_key": has_key,
            "requires_api_key": requires_key,
            "default_model": default_model,
            "base_url": settings.get_base_url(name),
        })

    return providers


def set_provider(provider: str, model: str | None = None) -> None:
    """Set the current LLM provider.

    Args:
        provider: Provider name.
        model: Optional model to set as current.

    Raises:
        ValueError: If provider is not registered.
    """
    if provider not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")

    settings = get_settings()
    settings.set_current_provider(provider)

    if model:
        settings.set_current_model(model)
    else:
        # Clear current model to use provider default
        settings.delete("current_model")


def register_provider(name: str, class_path: str, default_model: str = "") -> None:
    """Register a new LLM provider.

    Args:
        name: Provider name.
        class_path: Full import path to the LLM class.
        default_model: Default model for this provider.
    """
    PROVIDERS[name] = class_path
    if default_model:
        DEFAULT_MODELS[name] = default_model
