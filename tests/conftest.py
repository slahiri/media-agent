"""Pytest fixtures for image-gen tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample config file for testing."""
    config_content = """
models:
  image:
    pipeline:
      name: "test/image-model"
      torch_dtype: "float16"
      device: "cpu"
    split_files:
      repo_id: "test/split-model"
      text_encoder: "text_encoder.safetensors"
      diffusion_model: "diffusion.safetensors"
      vae: "vae.safetensors"
    local:
      text_encoder: "models/text_encoders/test.safetensors"
      diffusion_model: "models/diffusion_models/test.safetensors"
      vae: "models/vae/test.safetensors"
    default_steps: 8
    default_size: [512, 512]
    guidance_scale: 1.0
  llm:
    name: "test/llm-model"
    local_path: "models/llm/test-model"
    torch_dtype: "float16"
    device: "cpu"
    max_new_tokens: 100
  ocr:
    name: "test/ocr-model"
    torch_dtype: "float16"
    device: "cpu"
    quantization: "4bit"
    default_resolution: "gundam"

paths:
  cache_dir: "{cache_dir}"
  models_dir: "{models_dir}"
""".format(cache_dir=str(temp_dir / "cache"), models_dir=str(temp_dir / "models"))

    config_path = temp_dir / "config.yaml"
    config_path.write_text(config_content)

    # Create models directories
    (temp_dir / "models" / "text_encoders").mkdir(parents=True)
    (temp_dir / "models" / "diffusion_models").mkdir(parents=True)
    (temp_dir / "models" / "vae").mkdir(parents=True)
    (temp_dir / "models" / "llm").mkdir(parents=True)
    (temp_dir / "cache").mkdir(parents=True)

    return config_path


@pytest.fixture
def mock_pipeline():
    """Mock diffusers ZImagePipeline."""
    with patch("diffusers.ZImagePipeline") as mock:
        pipeline_instance = MagicMock()
        mock.from_pretrained.return_value = pipeline_instance

        # Mock image generation result
        mock_image = MagicMock()
        pipeline_instance.return_value.images = [mock_image]

        yield mock


@pytest.fixture
def mock_transformers():
    """Mock transformers AutoModelForCausalLM and AutoTokenizer."""
    with patch("transformers.AutoModelForCausalLM") as mock_model, \
         patch("transformers.AutoTokenizer") as mock_tokenizer:

        model_instance = MagicMock()
        tokenizer_instance = MagicMock()

        mock_model.from_pretrained.return_value = model_instance
        mock_tokenizer.from_pretrained.return_value = tokenizer_instance

        # Mock tokenizer behavior
        tokenizer_instance.return_value = {"input_ids": MagicMock()}
        tokenizer_instance.eos_token_id = 0
        tokenizer_instance.decode.return_value = "Generated text response"
        tokenizer_instance.apply_chat_template.return_value = "formatted chat"

        # Mock model behavior
        model_instance.device = "cpu"
        model_instance.generate.return_value = MagicMock()
        model_instance.generate.return_value.__getitem__ = lambda self, idx: MagicMock()

        yield {"model": mock_model, "tokenizer": mock_tokenizer}


@pytest.fixture
def mock_huggingface_hub():
    """Mock huggingface_hub download functions."""
    with patch("huggingface_hub.snapshot_download") as mock_snapshot, \
         patch("huggingface_hub.hf_hub_download") as mock_hf_download:

        mock_snapshot.return_value = "/fake/path/to/model"
        mock_hf_download.return_value = "/fake/path/to/file.safetensors"

        yield {
            "snapshot_download": mock_snapshot,
            "hf_hub_download": mock_hf_download,
        }
