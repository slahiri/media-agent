import os
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from image_gen.config import load_config, get_torch_dtype


class ImageGenerator:
    """Image generator using Z-Image-Turbo model."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        mode: Literal["pipeline", "local", "split"] = "pipeline",
        model_name: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
        enable_cpu_offload: bool = False,
    ):
        """Initialize the image generator.

        Args:
            config_path: Path to config file. Uses default if not provided.
            mode: Loading mode:
                - "pipeline": Load from HuggingFace using diffusers pipeline
                - "local": Load from local models folder (split files)
                - "split": Load split files from HuggingFace cache
            model_name: Override model name from config (pipeline mode only).
            device: Override device from config.
            torch_dtype: Override torch dtype from config.
            enable_cpu_offload: Enable CPU offload for memory-constrained devices.
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        image_config = self.config["models"]["image"]

        self.mode = mode
        self.model_name = model_name or image_config["pipeline"]["name"]
        self.device = device or image_config["pipeline"]["device"]
        self.torch_dtype = get_torch_dtype(torch_dtype or image_config["pipeline"]["torch_dtype"])
        self.default_steps = image_config.get("default_steps", 8)
        self.default_size = image_config.get("default_size", [1024, 1024])
        self.guidance_scale = image_config.get("guidance_scale", 1.0)
        self.enable_cpu_offload = enable_cpu_offload

        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy load the pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        if self.config_path:
            return Path(self.config_path).parent
        # Default to the package's parent directory
        return Path(__file__).parent.parent.parent

    def _get_local_model_paths(self) -> dict[str, Path]:
        """Get local model file paths."""
        local_config = self.config["models"]["image"]["local"]
        project_root = self._get_project_root()

        return {
            "text_encoder": project_root / local_config["text_encoder"],
            "diffusion_model": project_root / local_config["diffusion_model"],
            "vae": project_root / local_config["vae"],
        }

    def _check_local_models_exist(self) -> bool:
        """Check if all local model files exist."""
        paths = self._get_local_model_paths()
        return all(p.exists() for p in paths.values())

    def _load_pipeline(self):
        """Load the Z-Image pipeline."""
        from diffusers import ZImagePipeline

        cache_dir = self.config.get("paths", {}).get("cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)

        if self.mode == "pipeline":
            # Standard diffusers pipeline loading
            self._pipeline = ZImagePipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                cache_dir=cache_dir,
            )
        elif self.mode == "local":
            # Load from local split files
            if not self._check_local_models_exist():
                missing = [str(p) for p in self._get_local_model_paths().values() if not p.exists()]
                raise FileNotFoundError(
                    f"Local model files not found: {missing}. "
                    "Run download_image_model(mode='split', copy_to_local=True) first."
                )
            paths = self._get_local_model_paths()
            self._pipeline = self._load_from_split_files(paths)
        elif self.mode == "split":
            # Load from HuggingFace cache (split files)
            from image_gen.utils.downloader import get_split_file_paths
            paths = get_split_file_paths(self.config_path)
            self._pipeline = self._load_from_split_files(paths)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to(self.device)

    def _load_from_split_files(self, paths: dict[str, Path]) -> Any:
        """Load pipeline from split safetensor files."""
        from diffusers import ZImagePipeline, AutoencoderKL
        from diffusers.models import ZImageTransformer2DModel
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

        # This is a simplified loader - actual implementation depends on
        # how diffusers expects the components
        # For now, we'll use the standard pipeline and note this needs
        # further implementation based on diffusers internals

        # The safetensor files need to be loaded into the appropriate models
        # This typically requires knowing the exact model architecture

        # Fallback to pipeline mode with a warning
        print("Note: Split file loading requires specific model architecture setup.")
        print("Falling back to pipeline mode for now.")

        cache_dir = self.config.get("paths", {}).get("cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)

        return ZImagePipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            cache_dir=cache_dir,
        )

    def generate(
        self,
        prompt: str,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            height: Image height in pixels. Defaults to config value.
            width: Image width in pixels. Defaults to config value.
            num_inference_steps: Number of denoising steps. Defaults to config value.
            guidance_scale: Guidance scale. Defaults to config value.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            Generated PIL Image.
        """
        height = height or self.default_size[1]
        width = width or self.default_size[0]
        num_inference_steps = num_inference_steps or self.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        return result.images[0]

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Generate multiple images from text prompts.

        Args:
            prompts: List of text descriptions.
            **kwargs: Arguments passed to generate().

        Returns:
            List of generated PIL Images.
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
