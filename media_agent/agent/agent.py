"""LangGraph agent for image generation."""

from typing import Annotated, Optional, Literal, TypedDict, Sequence
from pathlib import Path
from datetime import datetime
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """State for the media agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class MediaAgent:
    """LangGraph agent for AI-powered image generation.

    Uses Qwen LLM for reasoning and Z-Image-Turbo for image generation.
    The agent interprets natural language requests and generates images.

    Example:
        >>> from media_agent import MediaAgent
        >>>
        >>> # Basic usage
        >>> agent = MediaAgent()
        >>> result = agent.run("Generate an image of a sunset over mountains")
        >>> print(result)  # "Image saved to: output/generated_xxx.png"
        >>>
        >>> # Multiple generations
        >>> agent.run("Create a cyberpunk city at night")
        >>> agent.run("A cat sitting on a beach")
        >>>
        >>> # Cleanup
        >>> agent.unload()

        >>> # Or use context manager
        >>> with MediaAgent() as agent:
        ...     agent.run("Generate a forest landscape")
    """

    def __init__(
        self,
        output_dir: str = "output",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        image_mode: str = "pipeline",
        offload_mode: str = "model",
        device: str = "cuda",
    ):
        """Initialize the media agent.

        Args:
            output_dir: Directory for generated images.
            llm_model: Qwen model name for reasoning.
            image_mode: Image model mode ("pipeline", "split", "local").
            offload_mode: GPU memory mode ("none", "model", "sequential").
            device: Device for models ("cuda" or "cpu").
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm_model = llm_model
        self.image_mode = image_mode
        self.offload_mode = offload_mode
        self.device = device

        # Lazy-loaded models
        self._llm = None
        self._generator = None
        self._graph = None

    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            from media_agent.llm.qwen import QwenLLM
            print(f"Loading LLM: {self.llm_model}")
            self._llm = QwenLLM(
                model_name=self.llm_model,
                device=self.device,
            )
        return self._llm

    @property
    def generator(self):
        """Lazy-load the image generator."""
        if self._generator is None:
            from media_agent.image.generator import ImageGenerator
            print(f"Loading ImageGenerator: mode={self.image_mode}")
            self._generator = ImageGenerator(
                mode=self.image_mode,
                offload_mode=self.offload_mode,
                device=self.device,
                keep_loaded=True,
            )
        return self._generator

    def _build_graph(self):
        """Build the LangGraph workflow."""
        if self._graph is not None:
            return self._graph

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_generate,
            {
                "generate": "generate",
                "end": END,
            }
        )
        workflow.add_edge("generate", "agent")

        self._graph = workflow.compile()
        return self._graph

    def _agent_node(self, state: AgentState) -> dict:
        """Agent node - LLM decides what to do."""
        messages = state["messages"]

        # Build chat messages
        chat_messages = [{"role": "system", "content": self._get_system_prompt()}]

        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                chat_messages.append({"role": "user", "content": f"Result: {msg.content}"})

        # Get LLM response
        response = self.llm.chat(chat_messages)

        # Parse for generation request
        gen_params = self._parse_generation(response)

        return {
            "messages": [AIMessage(
                content=response,
                additional_kwargs={"generation": gen_params} if gen_params else {}
            )]
        }

    def _generate_node(self, state: AgentState) -> dict:
        """Generate node - creates the image."""
        messages = state["messages"]
        last_message = messages[-1]

        gen_params = last_message.additional_kwargs.get("generation", {})

        try:
            # Generate image
            image = self.generator.generate(
                prompt=gen_params.get("prompt", ""),
                negative_prompt=gen_params.get("negative_prompt"),
                resolution=gen_params.get("resolution", "1024x1024"),
                seed=gen_params.get("seed"),
            )

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"
            output_path = self.output_dir / filename
            image.save(output_path)

            result = f"Image generated and saved to: {output_path}"

        except Exception as e:
            result = f"Error generating image: {str(e)}"

        return {"messages": [ToolMessage(content=result, tool_call_id="generate")]}

    def _should_generate(self, state: AgentState) -> Literal["generate", "end"]:
        """Decide if we should generate an image."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "additional_kwargs"):
            if last_message.additional_kwargs.get("generation"):
                return "generate"

        return "end"

    def _get_system_prompt(self) -> str:
        """System prompt for the agent."""
        return """You are an AI assistant that generates images from text descriptions.

When the user asks you to generate, create, draw, or make an image, respond with EXACTLY this format:

GENERATE_IMAGE
prompt: [detailed description of the image to generate]
negative_prompt: [optional - what to avoid, or "none"]
resolution: [optional - "1024x1024", "1344x768", "768x1344", or "none" for default]
seed: [optional - number for reproducibility, or "none"]

Examples:

User: "Create a sunset over mountains"
Response:
GENERATE_IMAGE
prompt: A breathtaking sunset over majestic mountain peaks, golden and orange sky, dramatic clouds, photorealistic
negative_prompt: blurry, low quality, distorted
resolution: 1344x768
seed: none

User: "Draw a cat"
Response:
GENERATE_IMAGE
prompt: A cute fluffy cat with bright eyes, sitting elegantly, soft lighting, detailed fur
negative_prompt: ugly, deformed
resolution: 1024x1024
seed: none

If the user is NOT asking for image generation, just respond conversationally.
If you just generated an image, summarize what you created."""

    def _parse_generation(self, response: str) -> Optional[dict]:
        """Parse generation parameters from LLM response."""
        if "GENERATE_IMAGE" not in response:
            return None

        params = {}
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("prompt:"):
                params["prompt"] = line[7:].strip()
            elif line.startswith("negative_prompt:"):
                val = line[16:].strip()
                if val.lower() not in ["none", ""]:
                    params["negative_prompt"] = val
            elif line.startswith("resolution:"):
                val = line[11:].strip()
                if val.lower() not in ["none", ""]:
                    params["resolution"] = val
            elif line.startswith("seed:"):
                val = line[5:].strip()
                if val.lower() not in ["none", ""]:
                    try:
                        params["seed"] = int(val)
                    except ValueError:
                        pass

        return params if params.get("prompt") else None

    def run(self, query: str) -> str:
        """Run the agent with a query.

        Args:
            query: Natural language request (e.g., "Generate a sunset image")

        Returns:
            Response string with result or generated image path.
        """
        graph = self._build_graph()

        result = graph.invoke({"messages": [HumanMessage(content=query)]})

        # Get final response
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                return msg.content
            elif isinstance(msg, AIMessage):
                if not msg.additional_kwargs.get("generation"):
                    return msg.content

        return "No response generated"

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        resolution: str = "1024x1024",
        seed: Optional[int] = None,
    ) -> str:
        """Generate an image directly (bypassing LLM).

        Args:
            prompt: Image description.
            negative_prompt: What to avoid.
            resolution: Image size ("1024x1024", "1344x768", etc.).
            seed: Random seed for reproducibility.

        Returns:
            Path to generated image.
        """
        image = self.generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            seed=seed,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        output_path = self.output_dir / filename
        image.save(output_path)

        return str(output_path)

    def unload(self):
        """Unload all models to free GPU memory."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
            print("ImageGenerator unloaded")

        if self._llm is not None:
            self._llm.unload()
            self._llm = None
            print("LLM unloaded")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False
