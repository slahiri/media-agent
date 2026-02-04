"""ASCII header component for the CLI."""

from rich.console import Console
from rich.text import Text


# ASCII art logo
LOGO = r"""
  __  __          _ _          _                    _
 |  \/  | ___  __| (_) __ _   / \   __ _  ___ _ __ | |_
 | |\/| |/ _ \/ _` | |/ _` | / _ \ / _` |/ _ \ '_ \| __|
 | |  | |  __/ (_| | | (_| |/ ___ \ (_| |  __/ | | | |_
 |_|  |_|\___|\__,_|_|\__,_/_/   \_\__, |\___|_| |_|\__|
                                   |___/
"""

# Compact logo for smaller terminals
LOGO_COMPACT = r"""
╔══════════════════════════════╗
║      M E D I A   A G E N T   ║
╚══════════════════════════════╝
"""

VERSION = "0.3.0"


def render_header(
    console: Console,
    compact: bool = False,
    show_version: bool = True,
) -> None:
    """Render the CLI header.

    Args:
        console: Rich Console instance.
        compact: Use compact logo for small terminals.
        show_version: Show version number.
    """
    logo = LOGO_COMPACT if compact else LOGO

    # Create styled logo
    logo_text = Text(logo, style="header")

    console.print(logo_text)

    if show_version:
        version_text = Text(f"v{VERSION}", style="header.accent")
        console.print(version_text, justify="center" if compact else "left")

    console.print()
