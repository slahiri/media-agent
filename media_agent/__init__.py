"""Media Utils - AI Agent for Image Generation.

This library provides a LangGraph-based agent that can generate images
from natural language prompts using Z-Image-Turbo and Qwen LLM.

Example:
    >>> from media_agent import MediaAgent
    >>>
    >>> agent = MediaAgent()
    >>> result = agent.run("Generate a sunset over mountains")
    >>> print(result)
    >>>
    >>> agent.unload()
"""

from media_agent.agent import MediaAgent

__version__ = "0.2.0"

__all__ = [
    "MediaAgent",
]
