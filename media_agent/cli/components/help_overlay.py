"""Help overlay component."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


COMMANDS_HELP = [
    # Chat
    ("Chat", [
        ("/help", "Show this help screen"),
        ("/clear", "Clear chat history"),
        ("/history", "Show conversation history"),
        ("/quit, /exit", "Exit the application"),
    ]),
    # Generation
    ("Image Generation", [
        ("/generate <prompt>", "Generate an image directly"),
        ("/resolutions", "List available image resolutions"),
        ("/unload", "Unload models to free GPU memory"),
    ]),
    # LLM
    ("LLM Providers", [
        ("/llm list", "List available LLM providers"),
        ("/llm <provider> [model]", "Switch LLM provider"),
        ("/llm openai gpt-4o", "Example: use OpenAI GPT-4o"),
        ("/llm anthropic claude-sonnet-4-20250514", "Example: use Claude"),
        ("/llm ollama llama3.2", "Example: use local Ollama"),
    ]),
    # Image Provider
    ("Image Provider", [
        ("/image list", "List available image providers"),
        ("/image local", "Use local Z-Image model"),
        ("/image nanobanana", "Use Nanobanana API"),
        ("/image local split", "Use local split files mode"),
    ]),
    # Settings
    ("Settings", [
        ("/settings list", "Show all settings"),
        ("/settings <key> <value>", "Set a configuration value"),
        ("/settings openai.api_key <key>", "Set OpenAI API key"),
        ("/settings anthropic.api_key <key>", "Set Anthropic API key"),
        ("/settings nanobanana.api_key <key>", "Set Nanobanana API key"),
    ]),
    # UI
    ("Interface", [
        ("/theme dark", "Switch to dark theme"),
        ("/theme light", "Switch to light theme"),
        ("?", "Show quick help"),
        ("Ctrl+C", "Cancel current operation"),
        ("↑/↓", "Navigate command history"),
        ("Tab", "Auto-complete commands"),
    ]),
]


def render_help(console: Console) -> None:
    """Render the help overlay.

    Args:
        console: Rich Console instance.
    """
    # Create tables for each section
    panels = []

    for section_name, commands in COMMANDS_HELP:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Command", style="command", no_wrap=True)
        table.add_column("Description", style="command.desc")

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        panels.append(Panel(
            table,
            title=f"[bold]{section_name}[/bold]",
            border_style="dim",
            padding=(0, 1),
        ))

    # Print all sections
    console.print()
    console.print("[bold]MediaAgent CLI Help[/bold]", justify="center")
    console.print()

    for panel in panels:
        console.print(panel)

    console.print()
    console.print("[dim]Press Enter to return to chat...[/dim]", justify="center")
    console.print()
