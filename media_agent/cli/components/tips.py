"""Getting started tips component."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


TIPS = [
    ("💡", "Type a message to chat with the AI about image generation"),
    ("🖼️", "Use /generate <prompt> to quickly generate an image"),
    ("⚙️", "Use /llm to switch between AI providers (OpenAI, Anthropic, etc.)"),
    ("🔑", "Use /settings to configure API keys"),
    ("📐", "Use /resolutions to see available image sizes"),
    ("❓", "Type /help or press ? for more commands"),
]


def render_tips(console: Console, show_all: bool = False) -> None:
    """Render getting started tips.

    Args:
        console: Rich Console instance.
        show_all: Show all tips (vs just first few).
    """
    tips_to_show = TIPS if show_all else TIPS[:4]

    lines = []
    for emoji, tip in tips_to_show:
        lines.append(f"  {emoji}  {tip}")

    content = "\n".join(lines)

    panel = Panel(
        content,
        title="[bold]Getting Started[/bold]",
        border_style="tips.border",
        padding=(0, 1),
    )

    console.print(panel)
    console.print()
