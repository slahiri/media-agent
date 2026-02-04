"""Main CLI application controller."""

from pathlib import Path
from typing import Any, Literal

from media_agent.cli.commands import CommandResult, execute_command
from media_agent.cli.components import render_header, render_help, render_tips
from media_agent.cli.input_handler import InputHandler
from media_agent.cli.progress import GenerationProgress, Spinner
from media_agent.cli.renderer import CLIRenderer
from media_agent.cli.themes import CLITheme, get_theme
from media_agent.image.generator import ImageGenerator
from media_agent.image.nanobanana import NanobananaGenerator
from media_agent.llm import get_llm, get_settings
from media_agent.llm.base import BaseLLM
from media_agent.tools.image_tool import get_tool_definitions, set_generator, set_output_dir


class CLIApp:
    """Main CLI application."""

    def __init__(
        self,
        output_dir: str | Path = "./outputs",
        theme: Literal["dark", "light"] = "dark",
        show_tips: bool = True,
    ):
        """Initialize the CLI application.

        Args:
            output_dir: Directory for generated images.
            theme: Color theme ("dark" or "light").
            show_tips: Show getting started tips on launch.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Theme and rendering
        self.theme = get_theme(theme)
        self.renderer = CLIRenderer(self.theme)
        self.input_handler = InputHandler()
        self.show_tips = show_tips

        # Progress display
        self.progress = GenerationProgress(self.renderer.console)

        # Chat history
        self.messages: list[dict[str, str]] = []

        # Models (lazy loaded)
        self._llm: BaseLLM | None = None
        self._image_generator: ImageGenerator | NanobananaGenerator | None = None

        # Settings
        self.settings = get_settings()

        # Configure image tool output directory
        set_output_dir(self.output_dir)

        # Set up input callbacks
        self.input_handler.on_help(self._show_help)

    @property
    def llm(self) -> BaseLLM:
        """Get the current LLM (lazy loaded)."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def image_generator(self) -> ImageGenerator | NanobananaGenerator:
        """Get the current image generator (lazy loaded)."""
        if self._image_generator is None:
            self._image_generator = self._create_image_generator()
            set_generator(self._image_generator)
        return self._image_generator

    def _create_image_generator(self) -> ImageGenerator | NanobananaGenerator:
        """Create the appropriate image generator based on settings."""
        provider = self.settings.get("image.provider", "local")

        if provider == "nanobanana":
            api_key = self.settings.get("nanobanana.api_key")
            if not api_key:
                raise ValueError(
                    "Nanobanana API key not set. Use: /settings nanobanana.api_key <key>"
                )
            base_url = self.settings.get("nanobanana.base_url")
            return NanobananaGenerator(
                api_key=api_key,
                base_url=base_url,
                output_dir=self.output_dir,
            )
        else:
            # Local generation
            mode = self.settings.get("image.mode", "pipeline")
            return ImageGenerator(mode=mode)

    def reload_llm(self) -> None:
        """Reload the LLM with current settings."""
        if self._llm is not None:
            self._llm.unload()
        self._llm = None

    def reload_image_generator(self) -> None:
        """Reload the image generator with current settings."""
        if self._image_generator is not None:
            self._image_generator.unload()
        self._image_generator = None
        # Force reload on next access
        set_generator(self.image_generator)

    def unload_models(self) -> None:
        """Unload all models to free GPU memory."""
        if self._llm is not None:
            self._llm.unload()
            self._llm = None
        if self._image_generator is not None:
            self._image_generator.unload()
            self._image_generator = None

    def set_theme(self, theme: Literal["dark", "light"]) -> None:
        """Change the color theme.

        Args:
            theme: Theme name.
        """
        self.theme = get_theme(theme)
        self.renderer.set_theme(self.theme)

    def clear_history(self) -> None:
        """Clear chat history."""
        self.messages = []

    def _show_help(self) -> None:
        """Show help overlay."""
        render_help(self.renderer.console)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are MediaAgent, an AI assistant specialized in image generation.
You help users create images by understanding their requests and generating appropriate images.

When the user asks you to generate an image:
1. Analyze their request and create a detailed prompt
2. Choose an appropriate resolution based on the content (landscape for wide scenes, portrait for tall subjects, square for balanced compositions)
3. Generate the image using the generate_image tool

Available resolutions:
- 1024x1024 (square) - balanced compositions
- 1344x768 (landscape 16:9) - wide scenes, landscapes
- 768x1344 (portrait 9:16) - tall subjects, portraits
- 1152x896 (landscape 4:3) - slight landscape
- 896x1152 (portrait 3:4) - slight portrait
- 1216x832, 832x1216 (3:2 ratio)
- 1536x640 (ultrawide) - panoramas

Be creative and helpful. If the user's request is vague, ask clarifying questions.
Always aim to create high-quality, detailed image prompts."""

    def _process_user_input(self, user_input: str) -> str | None:
        """Process user input and generate a response.

        Args:
            user_input: User's input text.

        Returns:
            Assistant response, or None if no response needed.
        """
        # Add to history
        self.messages.append({"role": "user", "content": user_input})

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            *self.messages,
        ]

        # Check if LLM supports tools
        if self.llm.supports_tools():
            return self._process_with_tools(messages)
        else:
            return self._process_without_tools(messages)

    def _process_with_tools(self, messages: list[dict]) -> str:
        """Process with tool-calling LLM.

        Args:
            messages: Chat messages.

        Returns:
            Assistant response.
        """
        tools = get_tool_definitions()

        with Spinner(self.renderer.console) as spinner:
            spinner.start("Thinking...")
            response = self.llm.chat_with_tools(messages, tools)

        # Handle tool calls
        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.name == "generate_image":
                    return self._handle_image_generation(tool_call.arguments)

        # Regular response
        self.messages.append({"role": "assistant", "content": response.content})
        return response.content

    def _process_without_tools(self, messages: list[dict]) -> str:
        """Process without tool-calling (parse structured output).

        Args:
            messages: Chat messages.

        Returns:
            Assistant response.
        """
        with Spinner(self.renderer.console) as spinner:
            spinner.start("Thinking...")
            response = self.llm.chat(messages)

        # Check for generation command in response
        if "GENERATE_IMAGE" in response:
            params = self._parse_generation_params(response)
            if params:
                return self._handle_image_generation(params)

        self.messages.append({"role": "assistant", "content": response})
        return response

    def _parse_generation_params(self, response: str) -> dict[str, Any] | None:
        """Parse generation parameters from structured response.

        Args:
            response: LLM response text.

        Returns:
            Dict of parameters or None.
        """
        if "GENERATE_IMAGE" not in response:
            return None

        params = {}
        lines = response.split("\n")
        in_block = False

        for line in lines:
            line = line.strip()
            if line == "GENERATE_IMAGE":
                in_block = True
                continue

            if in_block and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "prompt":
                    params["prompt"] = value
                elif key == "negative_prompt":
                    params["negative_prompt"] = value
                elif key == "resolution":
                    params["resolution"] = value
                elif key == "seed":
                    try:
                        params["seed"] = int(value)
                    except ValueError:
                        pass

        return params if "prompt" in params else None

    def _handle_image_generation(self, params: dict[str, Any]) -> str:
        """Handle image generation.

        Args:
            params: Generation parameters.

        Returns:
            Response message.
        """
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        resolution = params.get("resolution", "1024x1024")
        seed = params.get("seed")

        self.renderer.render_system_message(f"Generating: {prompt[:50]}...")

        try:
            # Start progress
            self.progress.start(8, "Generating image")

            # Generate image
            image = self.image_generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                resolution=resolution,
                seed=seed,
            )

            # Save image
            from datetime import datetime
            import uuid as uuid_mod

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid_mod.uuid4().hex[:8]
            filename = f"image_{timestamp}_{unique_id}.png"
            filepath = self.output_dir / filename
            image.save(filepath)

            # Finish progress
            self.progress.finish("Image generated")

            # Show preview
            self.renderer.render_image_preview(filepath)

            response = f"Generated image saved to: {filepath}"
            self.messages.append({"role": "assistant", "content": response})
            return response

        except Exception as e:
            self.progress.abort("Generation failed")
            error_msg = f"Failed to generate image: {e}"
            self.renderer.render_error(error_msg)
            return error_msg

    def run(self) -> None:
        """Run the main CLI loop."""
        # Show header
        render_header(self.renderer.console)

        # Show tips
        if self.show_tips:
            render_tips(self.renderer.console)

        # Main loop
        while True:
            try:
                # Get input
                user_input = self.input_handler.get_input("> ")

                if user_input is None:
                    # EOF or interrupt
                    break

                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    result = execute_command(self, user_input)

                    if result.clear_screen:
                        self.renderer.clear()
                        render_header(self.renderer.console)

                    if result.message:
                        if result.success:
                            self.renderer.render_system_message(result.message)
                        else:
                            self.renderer.render_error(result.message)

                    if result.exit_app:
                        break

                    # Handle special actions
                    if result.data and isinstance(result.data, dict):
                        action = result.data.get("action")
                        if action == "generate":
                            prompt = result.data.get("prompt", "")
                            self._handle_image_generation({"prompt": prompt})

                    continue

                # Regular chat input
                self.renderer.render_user_message(user_input)

                response = self._process_user_input(user_input)

                if response:
                    self.renderer.render_assistant_message(response)

            except KeyboardInterrupt:
                self.renderer.print()
                self.renderer.render_system_message("Use /quit to exit")
                continue

            except Exception as e:
                self.renderer.render_error(f"Error: {e}")
                continue

        # Cleanup
        self.renderer.render_system_message("Goodbye!")
        self.unload_models()


def run_cli(
    output_dir: str = "./outputs",
    theme: str = "dark",
    show_tips: bool = True,
) -> None:
    """Run the MediaAgent CLI.

    Args:
        output_dir: Directory for generated images.
        theme: Color theme ("dark" or "light").
        show_tips: Show getting started tips.
    """
    app = CLIApp(
        output_dir=output_dir,
        theme=theme,
        show_tips=show_tips,
    )
    app.run()
