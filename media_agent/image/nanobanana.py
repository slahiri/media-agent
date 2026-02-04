"""Nanobanana API client for cloud image generation."""

import io
import time
from pathlib import Path
from typing import Any, Callable
import uuid
from datetime import datetime

from PIL import Image

from media_agent.image.generator import RESOLUTIONS


class NanobananClient:
    """Client for Nanobanana image generation API."""

    DEFAULT_BASE_URL = "https://api.nanobanana.ai"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: int = 120,
    ):
        """Initialize the Nanobanana client.

        Args:
            api_key: Nanobanana API key.
            base_url: Optional base URL override.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self._session = None

    @property
    def session(self):
        """Lazy load the requests session."""
        if self._session is None:
            try:
                import requests
            except ImportError:
                raise ImportError(
                    "requests package not installed. Install with: pip install requests"
                )
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        return self._session

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        resolution: str = "1024x1024",
        seed: int | None = None,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        on_progress: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate an image using Nanobanana API.

        Args:
            prompt: Text description of the image.
            negative_prompt: What to avoid in the image.
            resolution: Image resolution (e.g., "1024x1024").
            seed: Random seed for reproducibility.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale.
            on_progress: Optional callback for progress updates.
            **kwargs: Additional API parameters.

        Returns:
            Generated PIL Image.
        """
        # Parse resolution
        if resolution in RESOLUTIONS:
            width, height = RESOLUTIONS[resolution]
        else:
            try:
                parts = resolution.lower().split("x")
                width, height = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                width, height = 1024, 1024

        # Build request payload
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if seed is not None:
            payload["seed"] = seed

        if on_progress:
            on_progress("Sending request to Nanobanana...")

        # Submit generation request
        response = self.session.post(
            f"{self.base_url}/v1/images/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()

        # Check if async (returns job_id)
        if "job_id" in result:
            return self._poll_job(result["job_id"], on_progress)

        # Sync response - image data directly
        if "image" in result:
            return self._decode_image(result["image"])

        if "url" in result:
            return self._download_image(result["url"], on_progress)

        raise ValueError(f"Unexpected API response: {result}")

    def _poll_job(
        self,
        job_id: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> Image.Image:
        """Poll for async job completion.

        Args:
            job_id: Job ID to poll.
            on_progress: Optional progress callback.

        Returns:
            Generated PIL Image.
        """
        max_polls = 60  # 2 minutes max
        poll_interval = 2  # seconds

        for i in range(max_polls):
            if on_progress:
                on_progress(f"Generating... ({i * poll_interval}s)")

            response = self.session.get(
                f"{self.base_url}/v1/jobs/{job_id}",
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            status = result.get("status", "").lower()

            if status == "completed":
                if "image" in result:
                    return self._decode_image(result["image"])
                if "url" in result:
                    return self._download_image(result["url"], on_progress)
                raise ValueError(f"Completed job has no image: {result}")

            if status == "failed":
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Generation failed: {error}")

            time.sleep(poll_interval)

        raise TimeoutError("Generation timed out")

    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image data.

        Args:
            image_data: Base64-encoded image.

        Returns:
            PIL Image.
        """
        import base64
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))

    def _download_image(
        self,
        url: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> Image.Image:
        """Download image from URL.

        Args:
            url: Image URL.
            on_progress: Optional progress callback.

        Returns:
            PIL Image.
        """
        if on_progress:
            on_progress("Downloading image...")

        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    def unload(self) -> None:
        """Close the session."""
        if self._session is not None:
            self._session.close()
            self._session = None


class NanobananaGenerator:
    """Wrapper to match ImageGenerator interface."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        output_dir: str | Path = "./outputs",
    ):
        """Initialize the Nanobanana generator.

        Args:
            api_key: Nanobanana API key.
            base_url: Optional base URL override.
            output_dir: Directory for saving images.
        """
        self.client = NanobananClient(api_key, base_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._on_progress: Callable[[str], None] | None = None

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set progress callback.

        Args:
            callback: Function called with status messages.
        """
        self._on_progress = callback

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        resolution: str = "1024x1024",
        seed: int | None = None,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate an image.

        Args:
            prompt: Text description.
            negative_prompt: What to avoid.
            resolution: Image size.
            seed: Random seed.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            **kwargs: Additional parameters.

        Returns:
            Generated PIL Image.
        """
        return self.client.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            on_progress=self._on_progress,
            **kwargs,
        )

    def generate_and_save(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        resolution: str = "1024x1024",
        seed: int | None = None,
        **kwargs: Any,
    ) -> Path:
        """Generate and save an image.

        Args:
            prompt: Text description.
            negative_prompt: What to avoid.
            resolution: Image size.
            seed: Random seed.
            **kwargs: Additional parameters.

        Returns:
            Path to saved image.
        """
        image = self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            seed=seed,
            **kwargs,
        )

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"image_{timestamp}_{unique_id}.png"
        filepath = self.output_dir / filename

        image.save(filepath)
        return filepath

    def unload(self) -> None:
        """Unload/close the client."""
        self.client.unload()

    @staticmethod
    def list_resolutions() -> dict[str, tuple[int, int]]:
        """List available resolutions."""
        return RESOLUTIONS.copy()
