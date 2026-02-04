"""DeepSeek-OCR module for document and image text extraction."""

from pathlib import Path
from typing import Optional, Union, Literal
import torch

from media_utils.config import load_config, get_torch_dtype


# OCR mode prompts
OCR_PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "free": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
}

# Resolution presets
RESOLUTIONS = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}


class DeepSeekOCR:
    """DeepSeek-OCR wrapper for document and image text extraction.

    Supports multiple quantization modes for different VRAM requirements:
    - None (full): ~16GB VRAM, best quality
    - "8bit": ~10-12GB VRAM, good quality
    - "4bit": ~8GB VRAM, acceptable quality

    Example:
        >>> ocr = DeepSeekOCR(quantization="4bit")
        >>> text = ocr.extract("document.png", mode="markdown")
        >>> print(text)
        >>> ocr.unload()
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantization: Optional[Literal["4bit", "8bit"]] = None,
        torch_dtype: Optional[str] = None,
        keep_loaded: bool = True,
    ):
        """Initialize DeepSeekOCR.

        Args:
            config_path: Path to config.yaml. If None, uses default.
            model_name: Override model name from config.
            device: Override device from config (e.g., "cuda:0").
            quantization: Quantization mode - None, "4bit", or "8bit".
            torch_dtype: Override torch dtype from config.
            keep_loaded: If True, keep model loaded after extraction.
        """
        self.config = load_config(config_path)
        ocr_config = self.config.get("models", {}).get("ocr", {})

        self.model_name = model_name or ocr_config.get("name", "deepseek-ai/DeepSeek-OCR-2")
        self.device = device or ocr_config.get("device", "cuda")
        self.quantization = quantization or ocr_config.get("quantization", None)
        self.keep_loaded = keep_loaded

        dtype_str = torch_dtype or ocr_config.get("torch_dtype", "bfloat16")
        self.torch_dtype = get_torch_dtype(dtype_str)

        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy-load tokenizer on first access."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading DeepSeek-OCR: {self.model_name}")
        print(f"  Quantization: {self.quantization or 'none (full precision)'}")
        print(f"  Device: {self.device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        load_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs["device_map"] = "auto"
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            # Full precision with Flash Attention
            load_kwargs["_attn_implementation"] = "flash_attention_2"

        self._model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        self._model = self._model.eval()

        # Move to GPU with bfloat16 for full precision mode
        if self.quantization is None:
            self._model = self._model.to(self.device).to(self.torch_dtype)

        print("DeepSeek-OCR loaded successfully")

    def extract(
        self,
        image: Union[str, Path, "Image.Image"],
        mode: Literal["markdown", "free", "figure", "describe"] = "markdown",
        resolution: Literal["tiny", "small", "base", "large", "gundam"] = "gundam",
        output_dir: Optional[Union[str, Path]] = None,
        save_results: bool = False,
    ) -> str:
        """Extract text from an image.

        Args:
            image: Path to image file or PIL Image.
            mode: Extraction mode:
                - "markdown": Convert document to markdown (default)
                - "free": Free-form OCR (text only)
                - "figure": Parse figures and charts
                - "describe": Describe image content
            resolution: Resolution preset affecting quality/speed:
                - "tiny": 512px, fastest, lowest quality
                - "small": 640px, fast
                - "base": 1024px, balanced
                - "large": 1280px, high quality
                - "gundam": 1024/640px with cropping (recommended for docs)
            output_dir: Directory to save results (if save_results=True).
            save_results: Whether to save results to files.

        Returns:
            Extracted text as string.
        """
        from PIL import Image

        # Handle image input
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            # PIL Image - save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name

        # Get prompt and resolution config
        prompt = OCR_PROMPTS.get(mode, OCR_PROMPTS["markdown"])
        res_config = RESOLUTIONS.get(resolution, RESOLUTIONS["gundam"])

        # Setup output directory
        if output_dir is None:
            output_dir = Path("./output/ocr")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run inference
        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=str(output_dir),
            base_size=res_config["base_size"],
            image_size=res_config["image_size"],
            crop_mode=res_config["crop_mode"],
            save_results=save_results,
            test_compress=False,
        )

        # Unload if not keeping loaded
        if not self.keep_loaded:
            self.unload()

        return result

    def extract_batch(
        self,
        images: list[Union[str, Path]],
        mode: Literal["markdown", "free", "figure", "describe"] = "markdown",
        resolution: Literal["tiny", "small", "base", "large", "gundam"] = "gundam",
    ) -> list[str]:
        """Extract text from multiple images.

        Args:
            images: List of image paths.
            mode: Extraction mode.
            resolution: Resolution preset.

        Returns:
            List of extracted texts.
        """
        results = []
        for image in images:
            result = self.extract(image, mode=mode, resolution=resolution)
            results.append(result)
        return results

    def unload(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("DeepSeek-OCR unloaded")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False

    @staticmethod
    def list_modes() -> list[str]:
        """List available OCR modes."""
        return list(OCR_PROMPTS.keys())

    @staticmethod
    def list_resolutions() -> list[str]:
        """List available resolution presets."""
        return list(RESOLUTIONS.keys())
