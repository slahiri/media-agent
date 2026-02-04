"""Input handling with prompt_toolkit for the CLI."""

from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


# Available slash commands
COMMANDS = [
    "/help",
    "/clear",
    "/quit",
    "/exit",
    "/llm",
    "/settings",
    "/generate",
    "/resolutions",
    "/theme",
    "/unload",
    "/history",
]

# Command arguments
COMMAND_ARGS = {
    "/llm": ["openai", "anthropic", "ollama", "huggingface", "list"],
    "/theme": ["dark", "light"],
    "/settings": ["openai.api_key", "anthropic.api_key", "ollama.base_url", "show", "list"],
}


class CommandCompleter(Completer):
    """Custom completer for slash commands."""

    def get_completions(self, document, complete_event):
        """Get completions for the current input."""
        text = document.text_before_cursor
        words = text.split()

        # Complete commands
        if text.startswith("/"):
            if len(words) == 1 and not text.endswith(" "):
                # Complete the command itself
                word = words[0]
                for cmd in COMMANDS:
                    if cmd.startswith(word):
                        yield Completion(
                            cmd,
                            start_position=-len(word),
                            display_meta=self._get_command_help(cmd),
                        )
            elif len(words) >= 1:
                # Complete command arguments
                cmd = words[0]
                if cmd in COMMAND_ARGS:
                    current_arg = words[-1] if not text.endswith(" ") else ""
                    for arg in COMMAND_ARGS[cmd]:
                        if arg.startswith(current_arg):
                            yield Completion(
                                arg,
                                start_position=-len(current_arg),
                            )

    def _get_command_help(self, cmd: str) -> str:
        """Get short help for a command."""
        help_map = {
            "/help": "Show help",
            "/clear": "Clear chat",
            "/quit": "Exit",
            "/exit": "Exit",
            "/llm": "Switch LLM provider",
            "/settings": "Configure settings",
            "/generate": "Generate image",
            "/resolutions": "List resolutions",
            "/theme": "Change theme",
            "/unload": "Free GPU memory",
            "/history": "Show history",
        }
        return help_map.get(cmd, "")


class InputHandler:
    """Handles user input with prompt_toolkit."""

    def __init__(
        self,
        history_file: str | Path | None = None,
        multiline: bool = False,
    ):
        """Initialize the input handler.

        Args:
            history_file: Path to history file. Defaults to ~/.media_agent_history.
            multiline: Enable multiline input mode.
        """
        # Set up history
        if history_file is None:
            history_file = Path.home() / ".media_agent_history"
        self.history_file = Path(history_file)
        self.history = FileHistory(str(self.history_file))

        # Set up key bindings
        self.bindings = self._create_key_bindings()

        # Create style
        self.style = Style.from_dict({
            "prompt": "bold green",
            "continuation": "dim green",
        })

        # Create completer
        self.completer = CommandCompleter()

        # Create session
        self.session: PromptSession = PromptSession(
            history=self.history,
            completer=self.completer,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings,
            style=self.style,
            multiline=multiline,
            enable_history_search=True,
        )

        # Callbacks
        self._on_help: Callable[[], None] | None = None
        self._on_interrupt: Callable[[], None] | None = None

    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings."""
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _(event):
            """Handle Ctrl+C."""
            if self._on_interrupt:
                self._on_interrupt()
            else:
                event.app.exit(exception=KeyboardInterrupt)

        @bindings.add("?")
        def _(event):
            """Show help on ?."""
            buffer = event.app.current_buffer
            # Only trigger help if ? is the only character
            if buffer.text == "":
                if self._on_help:
                    self._on_help()
                else:
                    buffer.insert_text("?")
            else:
                buffer.insert_text("?")

        return bindings

    def on_help(self, callback: Callable[[], None]) -> None:
        """Set callback for help request.

        Args:
            callback: Function to call when help is requested.
        """
        self._on_help = callback

    def on_interrupt(self, callback: Callable[[], None]) -> None:
        """Set callback for interrupt (Ctrl+C).

        Args:
            callback: Function to call on interrupt.
        """
        self._on_interrupt = callback

    def get_input(self, prompt: str = "> ") -> str | None:
        """Get input from the user.

        Args:
            prompt: Prompt string to display.

        Returns:
            User input string, or None if interrupted/EOF.
        """
        try:
            return self.session.prompt(prompt)
        except KeyboardInterrupt:
            return None
        except EOFError:
            return None

    def get_multiline_input(self, prompt: str = "> ") -> str | None:
        """Get multiline input from the user.

        Args:
            prompt: Prompt string to display.

        Returns:
            User input string, or None if interrupted/EOF.
        """
        try:
            return self.session.prompt(
                prompt,
                multiline=True,
                prompt_continuation="... ",
            )
        except KeyboardInterrupt:
            return None
        except EOFError:
            return None

    def get_password(self, prompt: str = "Password: ") -> str | None:
        """Get password input (hidden).

        Args:
            prompt: Prompt string to display.

        Returns:
            Password string, or None if interrupted/EOF.
        """
        try:
            return self.session.prompt(prompt, is_password=True)
        except KeyboardInterrupt:
            return None
        except EOFError:
            return None
