"""CLI commands for MediaAgent."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from media_agent.llm import get_settings, list_providers, set_provider
from media_agent.image.generator import RESOLUTIONS

if TYPE_CHECKING:
    from media_agent.cli.app import CLIApp


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    message: str = ""
    data: Any = None
    exit_app: bool = False
    clear_screen: bool = False


class CommandRegistry:
    """Registry for CLI commands."""

    def __init__(self):
        """Initialize the command registry."""
        self._commands: dict[str, Callable] = {}
        self._aliases: dict[str, str] = {}

    def register(self, name: str, handler: Callable, aliases: list[str] | None = None) -> None:
        """Register a command.

        Args:
            name: Command name (without /).
            handler: Command handler function.
            aliases: Optional list of aliases.
        """
        self._commands[name] = handler
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> Callable | None:
        """Get a command handler.

        Args:
            name: Command name or alias.

        Returns:
            Command handler or None.
        """
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]
        return self._commands.get(name)

    def list_commands(self) -> list[str]:
        """List all registered commands.

        Returns:
            List of command names.
        """
        return list(self._commands.keys())


# Global registry
_registry = CommandRegistry()


def command(name: str, aliases: list[str] | None = None):
    """Decorator to register a command.

    Args:
        name: Command name.
        aliases: Optional list of aliases.
    """
    def decorator(func: Callable) -> Callable:
        _registry.register(name, func, aliases)
        return func
    return decorator


def get_registry() -> CommandRegistry:
    """Get the command registry."""
    return _registry


# =============================================================================
# Commands
# =============================================================================


@command("help", aliases=["h", "?"])
def cmd_help(app: "CLIApp", args: list[str]) -> CommandResult:
    """Show help screen."""
    from media_agent.cli.components import render_help
    render_help(app.renderer.console)
    return CommandResult(success=True)


@command("quit", aliases=["exit", "q"])
def cmd_quit(app: "CLIApp", args: list[str]) -> CommandResult:
    """Exit the application."""
    return CommandResult(success=True, message="Goodbye!", exit_app=True)


@command("clear", aliases=["cls"])
def cmd_clear(app: "CLIApp", args: list[str]) -> CommandResult:
    """Clear the screen and chat history."""
    app.clear_history()
    return CommandResult(success=True, message="Chat cleared", clear_screen=True)


@command("history")
def cmd_history(app: "CLIApp", args: list[str]) -> CommandResult:
    """Show conversation history."""
    if not app.messages:
        return CommandResult(success=True, message="No messages in history")

    app.renderer.print_rule("Conversation History")
    for msg in app.messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            app.renderer.render_user_message(content)
        elif role == "assistant":
            app.renderer.render_assistant_message(content)
        elif role == "system":
            app.renderer.render_system_message(content)
    app.renderer.print_rule()

    return CommandResult(success=True)


@command("llm")
def cmd_llm(app: "CLIApp", args: list[str]) -> CommandResult:
    """Switch or list LLM providers."""
    if not args or args[0] == "list":
        # List providers
        providers = list_providers()
        lines = ["Available LLM providers:"]
        for p in providers:
            status = "✓" if p["has_api_key"] or not p["requires_api_key"] else "✗"
            current = " (current)" if p["is_current"] else ""
            key_status = " [needs API key]" if p["requires_api_key"] and not p["has_api_key"] else ""
            lines.append(f"  {status} {p['name']}{current}{key_status}")
            if p["default_model"]:
                lines.append(f"      default: {p['default_model']}")
        return CommandResult(success=True, message="\n".join(lines))

    # Switch provider
    provider = args[0]
    model = args[1] if len(args) > 1 else None

    try:
        set_provider(provider, model)
        app.reload_llm()
        msg = f"Switched to {provider}"
        if model:
            msg += f" ({model})"
        return CommandResult(success=True, message=msg)
    except ValueError as e:
        return CommandResult(success=False, message=str(e))


@command("image")
def cmd_image(app: "CLIApp", args: list[str]) -> CommandResult:
    """Switch or list image providers."""
    settings = get_settings()

    if not args or args[0] == "list":
        # List image providers
        current = settings.get("image.provider", "local")
        lines = ["Available image providers:"]

        providers_info = [
            ("local", "Local Z-Image model (GPU required)", False),
            ("nanobanana", "Nanobanana API (cloud)", True),
        ]

        for name, desc, needs_key in providers_info:
            status = "✓" if name == current else " "
            current_label = " (current)" if name == current else ""
            has_key = settings.get(f"{name}.api_key") is not None

            if needs_key and not has_key:
                key_status = " [needs API key]"
            else:
                key_status = ""

            lines.append(f"  [{status}] {name}{current_label}{key_status}")
            lines.append(f"       {desc}")

        # Show current mode for local
        if current == "local":
            mode = settings.get("image.mode", "pipeline")
            lines.append(f"\n  Local mode: {mode}")

        return CommandResult(success=True, message="\n".join(lines))

    # Switch provider
    provider = args[0].lower()

    if provider == "local":
        mode = args[1] if len(args) > 1 else "pipeline"
        if mode not in ("pipeline", "split", "local"):
            return CommandResult(
                success=False,
                message=f"Invalid mode: {mode}. Use 'pipeline', 'split', or 'local'."
            )
        settings.set("image.provider", "local")
        settings.set("image.mode", mode)
        app.reload_image_generator()
        return CommandResult(success=True, message=f"Switched to local image generation (mode: {mode})")

    elif provider == "nanobanana":
        api_key = settings.get("nanobanana.api_key")
        if not api_key:
            return CommandResult(
                success=False,
                message="Nanobanana API key not set. Use: /settings nanobanana.api_key <key>"
            )
        settings.set("image.provider", "nanobanana")
        app.reload_image_generator()
        return CommandResult(success=True, message="Switched to Nanobanana image generation")

    else:
        return CommandResult(
            success=False,
            message=f"Unknown image provider: {provider}. Use 'local' or 'nanobanana'."
        )


@command("settings")
def cmd_settings(app: "CLIApp", args: list[str]) -> CommandResult:
    """Configure settings."""
    settings = get_settings()

    if not args or args[0] in ("list", "show"):
        # Show all settings
        all_settings = settings.all_settings()

        if not all_settings:
            return CommandResult(success=True, message="No settings configured")

        lines = ["Current settings:"]
        for key, value in _flatten_dict(all_settings):
            # Mask API keys
            if "api_key" in key.lower() and value:
                value = value[:8] + "..." if len(str(value)) > 8 else "***"
            lines.append(f"  {key}: {value}")

        return CommandResult(success=True, message="\n".join(lines))

    if len(args) < 2:
        return CommandResult(
            success=False,
            message="Usage: /settings <key> <value>\n       /settings list"
        )

    key = args[0]
    value = " ".join(args[1:])

    # Set the value
    settings.set(key, value)

    # Mask display for API keys
    display_value = value
    if "api_key" in key.lower():
        display_value = value[:8] + "..." if len(value) > 8 else "***"

    return CommandResult(success=True, message=f"Set {key} = {display_value}")


@command("generate", aliases=["gen", "g"])
def cmd_generate(app: "CLIApp", args: list[str]) -> CommandResult:
    """Generate an image directly."""
    if not args:
        return CommandResult(
            success=False,
            message="Usage: /generate <prompt>\n       /generate a sunset over mountains"
        )

    prompt = " ".join(args)
    # The actual generation is handled by the app
    return CommandResult(success=True, data={"action": "generate", "prompt": prompt})


@command("resolutions", aliases=["res"])
def cmd_resolutions(app: "CLIApp", args: list[str]) -> CommandResult:
    """List available image resolutions."""
    lines = ["Available resolutions:"]
    for name, (w, h) in RESOLUTIONS.items():
        if w == h:
            ratio = "square"
        elif w > h:
            ratio = "landscape"
        else:
            ratio = "portrait"
        lines.append(f"  {name}: {w}x{h} ({ratio})")
    return CommandResult(success=True, message="\n".join(lines))


@command("theme")
def cmd_theme(app: "CLIApp", args: list[str]) -> CommandResult:
    """Change the CLI theme."""
    if not args:
        current = app.theme.name
        return CommandResult(
            success=True,
            message=f"Current theme: {current}\nUsage: /theme <dark|light>"
        )

    theme_name = args[0].lower()
    if theme_name not in ("dark", "light"):
        return CommandResult(
            success=False,
            message="Invalid theme. Use 'dark' or 'light'."
        )

    app.set_theme(theme_name)
    return CommandResult(success=True, message=f"Switched to {theme_name} theme")


@command("unload")
def cmd_unload(app: "CLIApp", args: list[str]) -> CommandResult:
    """Unload models to free GPU memory."""
    app.unload_models()
    return CommandResult(success=True, message="Models unloaded from GPU memory")


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a nested dict into key-value pairs with dot notation."""
    items = []
    for key, value in d.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, f"{full_key}."))
        else:
            items.append((full_key, value))
    return items


def execute_command(app: "CLIApp", command_line: str) -> CommandResult:
    """Execute a slash command.

    Args:
        app: CLIApp instance.
        command_line: Full command line (including /).

    Returns:
        CommandResult.
    """
    # Parse command
    parts = command_line[1:].split()  # Remove leading /
    if not parts:
        return CommandResult(success=False, message="Empty command")

    cmd_name = parts[0].lower()
    args = parts[1:]

    # Get handler
    handler = _registry.get(cmd_name)
    if handler is None:
        return CommandResult(
            success=False,
            message=f"Unknown command: /{cmd_name}\nType /help for available commands."
        )

    # Execute
    try:
        return handler(app, args)
    except Exception as e:
        return CommandResult(success=False, message=f"Command error: {e}")
