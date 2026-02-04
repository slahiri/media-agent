"""CLI themes for MediaAgent."""

from dataclasses import dataclass
from typing import Literal

from rich.style import Style
from rich.theme import Theme


@dataclass
class CLITheme:
    """Theme configuration for the CLI."""

    name: str

    # Message styles
    user_prefix: str
    user_text: str
    assistant_prefix: str
    assistant_text: str
    system_text: str
    error_text: str

    # UI elements
    header: str
    header_accent: str
    tips_border: str
    tips_text: str
    command: str
    command_desc: str

    # Progress
    progress_bar: str
    progress_complete: str
    progress_text: str

    # Prompt
    prompt: str
    prompt_continuation: str

    # Code and output
    code: str
    path: str
    dim: str
    highlight: str

    def to_rich_theme(self) -> Theme:
        """Convert to Rich Theme."""
        return Theme({
            "user.prefix": Style.parse(self.user_prefix),
            "user.text": Style.parse(self.user_text),
            "assistant.prefix": Style.parse(self.assistant_prefix),
            "assistant.text": Style.parse(self.assistant_text),
            "system": Style.parse(self.system_text),
            "error": Style.parse(self.error_text),
            "header": Style.parse(self.header),
            "header.accent": Style.parse(self.header_accent),
            "tips.border": Style.parse(self.tips_border),
            "tips.text": Style.parse(self.tips_text),
            "command": Style.parse(self.command),
            "command.desc": Style.parse(self.command_desc),
            "progress.bar": Style.parse(self.progress_bar),
            "progress.complete": Style.parse(self.progress_complete),
            "progress.text": Style.parse(self.progress_text),
            "prompt": Style.parse(self.prompt),
            "prompt.continuation": Style.parse(self.prompt_continuation),
            "code": Style.parse(self.code),
            "path": Style.parse(self.path),
            "dim": Style.parse(self.dim),
            "highlight": Style.parse(self.highlight),
        })


# Dark theme (default)
DARK_THEME = CLITheme(
    name="dark",
    # Messages
    user_prefix="bold green",
    user_text="white",
    assistant_prefix="bold magenta",
    assistant_text="white",
    system_text="dim cyan",
    error_text="bold red",
    # UI
    header="bold cyan",
    header_accent="bold magenta",
    tips_border="dim cyan",
    tips_text="dim white",
    command="bold yellow",
    command_desc="dim white",
    # Progress
    progress_bar="cyan",
    progress_complete="green",
    progress_text="dim white",
    # Prompt
    prompt="bold green",
    prompt_continuation="dim green",
    # Code
    code="dim cyan",
    path="underline blue",
    dim="dim",
    highlight="bold yellow",
)

# Light theme
LIGHT_THEME = CLITheme(
    name="light",
    # Messages
    user_prefix="bold blue",
    user_text="black",
    assistant_prefix="bold magenta",
    assistant_text="black",
    system_text="dim blue",
    error_text="bold red",
    # UI
    header="bold blue",
    header_accent="bold magenta",
    tips_border="dim blue",
    tips_text="dim black",
    command="bold cyan",
    command_desc="dim black",
    # Progress
    progress_bar="blue",
    progress_complete="green",
    progress_text="dim black",
    # Prompt
    prompt="bold blue",
    prompt_continuation="dim blue",
    # Code
    code="dim blue",
    path="underline cyan",
    dim="dim",
    highlight="bold cyan",
)


THEMES: dict[str, CLITheme] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}


def get_theme(name: Literal["dark", "light"] = "dark") -> CLITheme:
    """Get a theme by name.

    Args:
        name: Theme name ('dark' or 'light').

    Returns:
        CLITheme instance.
    """
    return THEMES.get(name, DARK_THEME)
