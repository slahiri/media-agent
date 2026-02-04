"""Settings management for LLM providers.

Stores API keys and configuration in a local JSON file instead of environment variables.
"""

import json
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path.home() / ".config" / "media_agent" / "settings.json"


class Settings:
    """Manage LLM provider settings and API keys."""

    def __init__(self, settings_path: Path | str | None = None):
        """Initialize settings manager.

        Args:
            settings_path: Path to settings file. Uses default if not provided.
        """
        self.settings_path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
        self._settings: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load settings from file."""
        if self.settings_path.exists():
            try:
                with open(self.settings_path) as f:
                    self._settings = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._settings = {}
        else:
            self._settings = {}

    def _save(self) -> None:
        """Save settings to file."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_path, "w") as f:
            json.dump(self._settings, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            key: Setting key (dot notation supported, e.g., 'openai.api_key').
            default: Default value if not found.

        Returns:
            Setting value or default.
        """
        keys = key.split(".")
        value = self._settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a setting value.

        Args:
            key: Setting key (dot notation supported).
            value: Value to set.
        """
        keys = key.split(".")
        current = self._settings
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        self._save()

    def delete(self, key: str) -> bool:
        """Delete a setting.

        Args:
            key: Setting key to delete.

        Returns:
            True if deleted, False if not found.
        """
        keys = key.split(".")
        current = self._settings
        for k in keys[:-1]:
            if k not in current:
                return False
            current = current[k]
        if keys[-1] in current:
            del current[keys[-1]]
            self._save()
            return True
        return False

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a provider.

        Args:
            provider: Provider name (openai, anthropic, etc.).

        Returns:
            API key or None.
        """
        return self.get(f"{provider}.api_key")

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider.

        Args:
            provider: Provider name.
            api_key: API key to store.
        """
        self.set(f"{provider}.api_key", api_key)

    def get_base_url(self, provider: str) -> str | None:
        """Get base URL for a provider.

        Args:
            provider: Provider name.

        Returns:
            Base URL or None.
        """
        return self.get(f"{provider}.base_url")

    def set_base_url(self, provider: str, base_url: str) -> None:
        """Set base URL for a provider.

        Args:
            provider: Provider name.
            base_url: Base URL to store.
        """
        self.set(f"{provider}.base_url", base_url)

    def get_default_model(self, provider: str) -> str | None:
        """Get default model for a provider.

        Args:
            provider: Provider name.

        Returns:
            Default model name or None.
        """
        return self.get(f"{provider}.default_model")

    def set_default_model(self, provider: str, model: str) -> None:
        """Set default model for a provider.

        Args:
            provider: Provider name.
            model: Model name to use as default.
        """
        self.set(f"{provider}.default_model", model)

    def get_current_provider(self) -> str:
        """Get the currently active provider.

        Returns:
            Provider name, defaults to 'huggingface'.
        """
        return self.get("current_provider", "huggingface")

    def set_current_provider(self, provider: str) -> None:
        """Set the currently active provider.

        Args:
            provider: Provider name to set as current.
        """
        self.set("current_provider", provider)

    def get_current_model(self) -> str | None:
        """Get the currently active model.

        Returns:
            Model name or None.
        """
        return self.get("current_model")

    def set_current_model(self, model: str) -> None:
        """Set the currently active model.

        Args:
            model: Model name to set as current.
        """
        self.set("current_model", model)

    def list_providers(self) -> dict[str, dict[str, Any]]:
        """List all configured providers.

        Returns:
            Dict of provider names to their settings.
        """
        providers = {}
        for key, value in self._settings.items():
            if isinstance(value, dict) and "api_key" in value:
                providers[key] = {
                    "has_api_key": bool(value.get("api_key")),
                    "base_url": value.get("base_url"),
                    "default_model": value.get("default_model"),
                }
        return providers

    def all_settings(self) -> dict[str, Any]:
        """Get all settings (for debugging).

        Returns:
            Copy of all settings.
        """
        return self._settings.copy()


# Global settings instance
_settings: Settings | None = None


def get_settings(settings_path: Path | str | None = None) -> Settings:
    """Get the global settings instance.

    Args:
        settings_path: Optional path override for settings file.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None or settings_path is not None:
        _settings = Settings(settings_path)
    return _settings
