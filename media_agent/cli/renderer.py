"""Rich-based message rendering for the CLI."""

from pathlib import Path
from typing import Literal

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .themes import CLITheme, get_theme


class CLIRenderer:
    """Renders CLI output using Rich."""

    def __init__(self, theme: CLITheme | None = None):
        """Initialize the renderer.

        Args:
            theme: Theme to use. Defaults to dark theme.
        """
        self.theme = theme or get_theme("dark")
        self.console = Console(theme=self.theme.to_rich_theme())

    def set_theme(self, theme: CLITheme | Literal["dark", "light"]) -> None:
        """Change the theme.

        Args:
            theme: Theme or theme name.
        """
        if isinstance(theme, str):
            theme = get_theme(theme)
        self.theme = theme
        self.console = Console(theme=self.theme.to_rich_theme())

    def render_user_message(self, content: str) -> None:
        """Render a user message.

        Args:
            content: Message content.
        """
        prefix = Text("> ", style="user.prefix")
        text = Text(content, style="user.text")
        self.console.print(prefix + text)

    def render_assistant_message(self, content: str) -> None:
        """Render an assistant message.

        Args:
            content: Message content (may contain markdown).
        """
        prefix = Text("◆ ", style="assistant.prefix")
        self.console.print(prefix, end="")

        # Render as markdown if it contains markdown elements
        if any(c in content for c in ["```", "#", "*", "_", "[", "`"]):
            md = Markdown(content)
            self.console.print(md)
        else:
            self.console.print(content, style="assistant.text")

    def render_system_message(self, content: str) -> None:
        """Render a system message.

        Args:
            content: Message content.
        """
        self.console.print(f"ℹ {content}", style="system")

    def render_error(self, message: str) -> None:
        """Render an error message.

        Args:
            message: Error message.
        """
        self.console.print(f"✗ {message}", style="error")

    def render_success(self, message: str) -> None:
        """Render a success message.

        Args:
            message: Success message.
        """
        self.console.print(f"✓ {message}", style="progress.complete")

    def render_image_preview(self, image_path: str | Path, clickable: bool = True) -> None:
        """Render an image preview with clickable link.

        Args:
            image_path: Path to the image file.
            clickable: Whether to make it a clickable hyperlink.
        """
        path = Path(image_path)
        abs_path = path.absolute()

        # Create file:// URL for terminal clickability
        file_url = f"file://{abs_path}"

        # Render image info
        self.console.print()
        self.console.print("🖼  Image generated:", style="progress.complete")

        if clickable:
            # Use Rich's built-in hyperlink support
            link = Text(f"  📎 {path.name}", style="path")
            link.stylize(f"link {file_url}")
            self.console.print(link)
            self.console.print(f"     {abs_path}", style="dim")
        else:
            self.console.print(f"  {abs_path}", style="path")

        # Try to display inline preview if terminal supports it (iTerm2, Kitty, etc.)
        self._try_inline_image(abs_path)

        self.console.print()

    def _try_inline_image(self, image_path: Path) -> None:
        """Try to display inline image preview if terminal supports it.

        Args:
            image_path: Path to the image file.
        """
        # Try using rich-pixels for inline images (if available)
        try:
            from rich_pixels import Pixels
            from PIL import Image

            # Load and resize image for preview
            img = Image.open(image_path)
            # Scale down for terminal display (max 60 chars wide)
            max_width = 60
            aspect = img.width / img.height
            preview_width = min(max_width, img.width // 10)
            preview_height = int(preview_width / aspect / 2)  # /2 for terminal char aspect

            pixels = Pixels.from_image_path(str(image_path), resize=(preview_width, preview_height))
            self.console.print(pixels)
        except ImportError:
            # rich-pixels not available, try kitty/iterm2 protocol
            self._try_native_image_protocol(image_path)
        except Exception:
            # Fall back to no preview
            pass

    def _try_native_image_protocol(self, image_path: Path) -> None:
        """Try native terminal image protocols (Kitty, iTerm2).

        Args:
            image_path: Path to the image file.
        """
        import os
        import sys
        import base64

        term = os.environ.get("TERM", "")
        term_program = os.environ.get("TERM_PROGRAM", "")

        # Check for Kitty
        if "kitty" in term.lower() or os.environ.get("KITTY_WINDOW_ID"):
            try:
                with open(image_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                # Kitty graphics protocol
                sys.stdout.write(f"\033_Ga=T,f=100,t=f;{data}\033\\")
                sys.stdout.flush()
                return
            except Exception:
                pass

        # Check for iTerm2
        if term_program == "iTerm.app" or os.environ.get("ITERM_SESSION_ID"):
            try:
                with open(image_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                # iTerm2 inline image protocol
                name = base64.b64encode(image_path.name.encode()).decode("ascii")
                sys.stdout.write(f"\033]1337;File=name={name};inline=1;width=auto;height=auto:{data}\007")
                sys.stdout.flush()
                return
            except Exception:
                pass

    def render_code(self, code: str, language: str = "python") -> None:
        """Render a code block.

        Args:
            code: Code content.
            language: Programming language for syntax highlighting.
        """
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)

    def render_panel(
        self,
        content: str,
        title: str = "",
        border_style: str = "dim",
    ) -> None:
        """Render content in a panel.

        Args:
            content: Panel content.
            title: Panel title.
            border_style: Border style.
        """
        panel = Panel(content, title=title, border_style=border_style)
        self.console.print(panel)

    def clear(self) -> None:
        """Clear the terminal screen."""
        self.console.clear()

    def print(self, *args, **kwargs) -> None:
        """Print with Rich formatting.

        Args:
            *args: Positional arguments for Console.print.
            **kwargs: Keyword arguments for Console.print.
        """
        self.console.print(*args, **kwargs)

    def print_rule(self, title: str = "") -> None:
        """Print a horizontal rule.

        Args:
            title: Optional title in the center of the rule.
        """
        self.console.rule(title, style="dim")
