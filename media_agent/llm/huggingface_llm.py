"""HuggingFace LLM provider using local transformers models."""

import os
from pathlib import Path
from typing import Any

from media_agent.config import get_torch_dtype, load_config

from .base import BaseLLM, LLMResponse, ToolCall


class HuggingFaceLLM(BaseLLM):
    """HuggingFace LLM using local transformers models (Qwen, Llama, etc.)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str | None = None,  # Not used, for interface compatibility
        base_url: str | None = None,  # Not used
        local_path: str | Path | None = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        config_path: str | Path | None = None,
        **kwargs: Any,
    ):
        """Initialize the HuggingFace LLM.

        Args:
            model: HuggingFace model name.
            api_key: Not used (for interface compatibility).
            base_url: Not used (for interface compatibility).
            local_path: Load from local path instead of HuggingFace.
            device: Device to load model on.
            torch_dtype: Torch dtype for model.
            config_path: Path to config file for default settings.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

        self.local_path = local_path
        self.device = device
        self.dtype = get_torch_dtype(torch_dtype)
        self.config_path = config_path

        # Load config for defaults
        try:
            self.config = load_config(config_path)
            llm_config = self.config.get("models", {}).get("llm", {})
            self.max_new_tokens = llm_config.get("max_new_tokens", 512)

            # Check for local path in config if not provided
            if self.local_path is None and "local_path" in llm_config:
                project_root = self._get_project_root()
                potential_local = project_root / llm_config["local_path"]
                if potential_local.exists():
                    self.local_path = str(potential_local)
        except Exception:
            self.config = {}
            self.max_new_tokens = 512

        self._model = None
        self._tokenizer = None

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        if self.config_path:
            return Path(self.config_path).parent
        return Path(__file__).parent.parent.parent

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "huggingface"

    @property
    def _hf_model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = self.config.get("paths", {}).get("cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)

        # Use local path if available, otherwise use HuggingFace
        model_path = self.local_path if self.local_path else self.model

        print(f"Loading LLM from: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir if not self.local_path else None,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            cache_dir=cache_dir if not self.local_path else None,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt."""
        top_p = kwargs.pop("top_p", 0.9)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)

        outputs = self._hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response in chat format."""
        top_p = kwargs.pop("top_p", 0.9)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self._hf_model.device)

        outputs = self._hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def supports_tools(self) -> bool:
        """HuggingFace models don't natively support tool calling."""
        return False

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with tool calling support.

        Note: HuggingFace models don't natively support tool calling,
        so we return a regular response without tool calls.
        """
        content = self.chat(messages, max_tokens, temperature, **kwargs)
        return LLMResponse(content=content, tool_calls=[])

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Clear CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None
