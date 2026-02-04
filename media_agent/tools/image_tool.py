"""LangChain tool for image generation using Z-Image."""

from pathlib import Path
from typing import Any, Callable
import uuid
from datetime import datetime

from langchain_core.tools import tool

from media_agent.image.generator import ImageGenerator, RESOLUTIONS


# Global generator instance (singleton pattern)
_generator: ImageGenerator | None = None
_output_dir: Path = Path("./outputs")
_step_callback: Callable[[int, int, float], None] | None = None


def set_generator(generator: ImageGenerator) -> None:
    """Set the global generator instance.

    Args:
        generator: ImageGenerator instance to use.
    """
    global _generator
    _generator = generator


def set_output_dir(output_dir: str | Path) -> None:
    """Set the output directory for generated images.

    Args:
        output_dir: Directory path for saving images.
    """
    global _output_dir
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)


def set_step_callback(callback: Callable[[int, int, float], None] | None) -> None:
    """Set a callback for generation progress.

    Args:
        callback: Function called with (current_step, total_steps, time_elapsed).
    """
    global _step_callback
    _step_callback = callback


def get_generator() -> ImageGenerator:
    """Get or create the global generator instance.

    Returns:
        ImageGenerator instance.
    """
    global _generator
    if _generator is None:
        _generator = ImageGenerator()
    return _generator


@tool
def generate_image(
    prompt: str,
    negative_prompt: str = "",
    resolution: str = "1024x1024",
    seed: int | None = None,
) -> str:
    """Generate an image from a text description.

    This tool creates high-quality images from natural language descriptions
    using the Z-Image-Turbo model. Use detailed, descriptive prompts for
    best results.

    Args:
        prompt: Detailed description of the image to generate. Be specific
            about subjects, style, lighting, colors, and composition.
        negative_prompt: What to avoid in the image (e.g., "blurry, low quality").
            Leave empty if not needed.
        resolution: Image size. Options:
            - "1024x1024" (square, default)
            - "1344x768" (landscape 16:9)
            - "768x1344" (portrait 9:16)
            - "1152x896" (landscape 4:3)
            - "896x1152" (portrait 3:4)
            - "1216x832" (landscape 3:2)
            - "832x1216" (portrait 2:3)
            - "1536x640" (ultrawide 21:9)
            - "640x1536" (tall 9:21)
        seed: Random seed for reproducibility. Use the same seed to get
            identical results. Leave as None for random generation.

    Returns:
        Path to the generated image file.
    """
    global _generator, _output_dir, _step_callback

    generator = get_generator()

    # Ensure output directory exists
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the image
    image = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        resolution=resolution,
        seed=seed,
    )

    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"image_{timestamp}_{unique_id}.png"
    filepath = _output_dir / filename

    # Save the image
    image.save(filepath)

    return str(filepath)


@tool
def list_resolutions() -> str:
    """List all available image resolution presets.

    Returns:
        Formatted list of available resolutions with dimensions.
    """
    lines = ["Available resolutions:"]
    for name, (w, h) in RESOLUTIONS.items():
        aspect = f"{w}x{h}"
        if w == h:
            ratio = "1:1 (square)"
        elif w > h:
            ratio = f"landscape ({w//h if w % h == 0 else round(w/h, 2)}:1)"
        else:
            ratio = f"portrait (1:{h//w if h % w == 0 else round(h/w, 2)})"
        lines.append(f"  {name}: {aspect} - {ratio}")
    return "\n".join(lines)


def get_image_tools() -> list:
    """Get all image-related LangChain tools.

    Returns:
        List of tool functions.
    """
    return [generate_image, list_resolutions]


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get OpenAI-format tool definitions for the image tools.

    Returns:
        List of tool definition dicts.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image from a text description using Z-Image-Turbo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the image to generate.",
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "What to avoid in the image.",
                            "default": "",
                        },
                        "resolution": {
                            "type": "string",
                            "enum": list(RESOLUTIONS.keys()),
                            "description": "Image resolution preset.",
                            "default": "1024x1024",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility.",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_resolutions",
                "description": "List all available image resolution presets.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]
