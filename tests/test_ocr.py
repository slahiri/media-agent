"""Tests for DeepSeekOCR class."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from media_utils.ocr.deepseek import DeepSeekOCR, OCR_PROMPTS, RESOLUTIONS


class TestDeepSeekOCRInit:
    """Tests for DeepSeekOCR initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with config path."""
        ocr = DeepSeekOCR(config_path=sample_config)

        assert ocr.model_name == "test/ocr-model"
        assert ocr.device == "cpu"
        assert ocr.quantization == "4bit"

    def test_init_with_model_name_override(self, sample_config):
        """Test initialization with model name override."""
        ocr = DeepSeekOCR(config_path=sample_config, model_name="custom/ocr")
        assert ocr.model_name == "custom/ocr"

    def test_init_with_device_override(self, sample_config):
        """Test initialization with device override."""
        ocr = DeepSeekOCR(config_path=sample_config, device="cuda:0")
        assert ocr.device == "cuda:0"

    def test_init_with_quantization_override(self, sample_config):
        """Test initialization with quantization override."""
        ocr = DeepSeekOCR(config_path=sample_config, quantization="8bit")
        assert ocr.quantization == "8bit"

    def test_init_lazy_load(self, sample_config):
        """Test that model is not loaded on init."""
        ocr = DeepSeekOCR(config_path=sample_config)
        assert ocr._model is None
        assert ocr._tokenizer is None

    def test_init_keep_loaded_default(self, sample_config):
        """Test keep_loaded defaults to True."""
        ocr = DeepSeekOCR(config_path=sample_config)
        assert ocr.keep_loaded is True

    def test_init_keep_loaded_false(self, sample_config):
        """Test keep_loaded can be set to False."""
        ocr = DeepSeekOCR(config_path=sample_config, keep_loaded=False)
        assert ocr.keep_loaded is False


class TestDeepSeekOCRModelLoading:
    """Tests for DeepSeekOCR model loading."""

    def test_model_property_triggers_load(self, sample_config):
        """Test that accessing model property triggers loading."""
        ocr = DeepSeekOCR(config_path=sample_config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("transformers.AutoModel") as mock_model_class, \
             patch("transformers.AutoTokenizer") as mock_tokenizer_class:
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Access model property
            _ = ocr.model

            # Model should be loaded
            mock_model_class.from_pretrained.assert_called_once()

    def test_tokenizer_property_triggers_load(self, sample_config):
        """Test that accessing tokenizer property triggers loading."""
        ocr = DeepSeekOCR(config_path=sample_config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("transformers.AutoModel") as mock_model_class, \
             patch("transformers.AutoTokenizer") as mock_tokenizer_class:
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Access tokenizer property
            _ = ocr.tokenizer

            # Tokenizer should be loaded
            mock_tokenizer_class.from_pretrained.assert_called_once()


class TestDeepSeekOCRExtract:
    """Tests for DeepSeekOCR.extract method."""

    def _setup_mocks(self, ocr):
        """Setup mock model and tokenizer for testing."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Mock infer method
        mock_model.infer.return_value = "Extracted text from image"

        ocr._model = mock_model
        ocr._tokenizer = mock_tokenizer

        return mock_model, mock_tokenizer

    def test_extract_basic(self, sample_config, temp_dir):
        """Test basic text extraction."""
        ocr = DeepSeekOCR(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(ocr)

        # Create a fake image file
        image_path = temp_dir / "test_image.png"
        image_path.touch()

        result = ocr.extract(str(image_path))

        assert result == "Extracted text from image"
        mock_model.infer.assert_called_once()

    def test_extract_with_mode(self, sample_config, temp_dir):
        """Test extraction with different modes."""
        ocr = DeepSeekOCR(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(ocr)

        image_path = temp_dir / "test_image.png"
        image_path.touch()

        for mode in ["markdown", "free", "figure", "describe"]:
            result = ocr.extract(str(image_path), mode=mode)

            call_args = mock_model.infer.call_args
            prompt_used = call_args[1]["prompt"]
            assert prompt_used == OCR_PROMPTS[mode]

    def test_extract_with_resolution(self, sample_config, temp_dir):
        """Test extraction with different resolutions."""
        ocr = DeepSeekOCR(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(ocr)

        image_path = temp_dir / "test_image.png"
        image_path.touch()

        for resolution in ["tiny", "small", "base", "large", "gundam"]:
            result = ocr.extract(str(image_path), resolution=resolution)

            call_args = mock_model.infer.call_args
            res_config = RESOLUTIONS[resolution]
            assert call_args[1]["base_size"] == res_config["base_size"]
            assert call_args[1]["image_size"] == res_config["image_size"]
            assert call_args[1]["crop_mode"] == res_config["crop_mode"]

    def test_extract_unloads_when_keep_loaded_false(self, sample_config, temp_dir):
        """Test that model unloads after extraction when keep_loaded=False."""
        ocr = DeepSeekOCR(config_path=sample_config, keep_loaded=False)
        mock_model, mock_tokenizer = self._setup_mocks(ocr)

        image_path = temp_dir / "test_image.png"
        image_path.touch()

        with patch.object(ocr, 'unload') as mock_unload:
            result = ocr.extract(str(image_path))
            mock_unload.assert_called_once()


class TestDeepSeekOCRBatch:
    """Tests for DeepSeekOCR.extract_batch method."""

    def test_extract_batch(self, sample_config, temp_dir):
        """Test batch extraction."""
        ocr = DeepSeekOCR(config_path=sample_config)

        mock_model = MagicMock()
        mock_model.infer.side_effect = ["Text 1", "Text 2", "Text 3"]
        ocr._model = mock_model
        ocr._tokenizer = MagicMock()

        # Create fake image files
        images = []
        for i in range(3):
            image_path = temp_dir / f"image_{i}.png"
            image_path.touch()
            images.append(str(image_path))

        results = ocr.extract_batch(images)

        assert len(results) == 3
        assert results == ["Text 1", "Text 2", "Text 3"]
        assert mock_model.infer.call_count == 3


class TestDeepSeekOCRContextManager:
    """Tests for DeepSeekOCR context manager."""

    def test_context_manager_unloads(self, sample_config):
        """Test that context manager unloads model on exit."""
        with patch.object(DeepSeekOCR, 'unload') as mock_unload:
            with DeepSeekOCR(config_path=sample_config) as ocr:
                pass
            mock_unload.assert_called_once()


class TestDeepSeekOCRStaticMethods:
    """Tests for DeepSeekOCR static methods."""

    def test_list_modes(self):
        """Test list_modes returns all modes."""
        modes = DeepSeekOCR.list_modes()
        assert set(modes) == {"markdown", "free", "figure", "describe"}

    def test_list_resolutions(self):
        """Test list_resolutions returns all resolutions."""
        resolutions = DeepSeekOCR.list_resolutions()
        assert set(resolutions) == {"tiny", "small", "base", "large", "gundam"}
