"""Progress display for image generation."""

import time
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


class GenerationProgress:
    """Display progress for image generation."""

    def __init__(self, console: Console | None = None):
        """Initialize the progress display.

        Args:
            console: Rich Console instance to use.
        """
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id = None
        self._start_time: float = 0

    def start(self, total_steps: int, description: str = "Generating") -> None:
        """Start the progress display.

        Args:
            total_steps: Total number of steps.
            description: Description text.
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="progress.complete", finished_style="progress.complete"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(description, total=total_steps)
        self._start_time = time.time()

    def update(self, step: int, description: str | None = None) -> None:
        """Update the progress.

        Args:
            step: Current step number.
            description: Optional new description.
        """
        if self._progress and self._task_id is not None:
            update_kwargs = {"completed": step}
            if description:
                update_kwargs["description"] = description
            self._progress.update(self._task_id, **update_kwargs)

    def finish(self, message: str = "Done") -> float:
        """Finish the progress display.

        Args:
            message: Completion message.

        Returns:
            Total elapsed time in seconds.
        """
        elapsed = time.time() - self._start_time

        if self._progress:
            self._progress.stop()
            self._progress = None

        self.console.print(f"✓ {message} ({elapsed:.1f}s)", style="bold green")

        return elapsed

    def abort(self, message: str = "Aborted") -> None:
        """Abort the progress display.

        Args:
            message: Abort message.
        """
        if self._progress:
            self._progress.stop()
            self._progress = None

        self.console.print(f"✗ {message}", style="bold red")

    def get_callback(self) -> Callable[[int, int, float], None]:
        """Get a callback function for step updates.

        Returns:
            Callback function that takes (current_step, total_steps, time_elapsed).
        """
        def callback(current_step: int, total_steps: int, time_elapsed: float) -> None:
            self.update(current_step)

        return callback


class Spinner:
    """Simple spinner for indeterminate operations."""

    def __init__(self, console: Console | None = None):
        """Initialize the spinner.

        Args:
            console: Rich Console instance to use.
        """
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id = None

    def start(self, message: str = "Processing...") -> None:
        """Start the spinner.

        Args:
            message: Message to display.
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(message, total=None)

    def update(self, message: str) -> None:
        """Update the spinner message.

        Args:
            message: New message to display.
        """
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=message)

    def stop(self) -> None:
        """Stop the spinner."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
