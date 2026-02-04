"""Tests for the MediaAgent."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from media_utils.agent.agent import MediaAgent


class TestMediaAgentInit:
    """Tests for MediaAgent initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        agent = MediaAgent()

        assert agent.output_dir == Path("output")
        assert agent.llm_model == "Qwen/Qwen2.5-7B-Instruct"
        assert agent.image_mode == "pipeline"
        assert agent.offload_mode == "model"
        assert agent.device == "cuda"

    def test_init_custom_params(self, temp_dir):
        """Test initialization with custom parameters."""
        agent = MediaAgent(
            output_dir=str(temp_dir),
            llm_model="custom/model",
            image_mode="split",
            offload_mode="sequential",
            device="cpu",
        )

        assert agent.output_dir == temp_dir
        assert agent.llm_model == "custom/model"
        assert agent.image_mode == "split"
        assert agent.offload_mode == "sequential"
        assert agent.device == "cpu"

    def test_init_creates_output_dir(self, temp_dir):
        """Test that output directory is created."""
        output_dir = temp_dir / "new_output"
        agent = MediaAgent(output_dir=str(output_dir))

        assert output_dir.exists()

    def test_lazy_loading(self):
        """Test that models are not loaded on init."""
        agent = MediaAgent()

        assert agent._llm is None
        assert agent._generator is None


class TestMediaAgentParsing:
    """Tests for response parsing."""

    def test_parse_generation_valid(self):
        """Test parsing valid generation response."""
        agent = MediaAgent()

        response = """GENERATE_IMAGE
prompt: A beautiful sunset
negative_prompt: blurry
resolution: 1024x1024
seed: 42"""

        result = agent._parse_generation(response)

        assert result is not None
        assert result["prompt"] == "A beautiful sunset"
        assert result["negative_prompt"] == "blurry"
        assert result["resolution"] == "1024x1024"
        assert result["seed"] == 42

    def test_parse_generation_minimal(self):
        """Test parsing with only prompt."""
        agent = MediaAgent()

        response = """GENERATE_IMAGE
prompt: Just a simple prompt
negative_prompt: none
resolution: none
seed: none"""

        result = agent._parse_generation(response)

        assert result is not None
        assert result["prompt"] == "Just a simple prompt"
        assert "negative_prompt" not in result
        assert "resolution" not in result
        assert "seed" not in result

    def test_parse_generation_no_match(self):
        """Test parsing non-generation response."""
        agent = MediaAgent()

        response = "This is just a normal conversation response."

        result = agent._parse_generation(response)

        assert result is None

    def test_parse_generation_no_prompt(self):
        """Test parsing with missing prompt."""
        agent = MediaAgent()

        response = """GENERATE_IMAGE
negative_prompt: blurry
resolution: 1024x1024"""

        result = agent._parse_generation(response)

        assert result is None


class TestMediaAgentSystemPrompt:
    """Tests for system prompt."""

    def test_system_prompt_content(self):
        """Test system prompt contains required info."""
        agent = MediaAgent()

        prompt = agent._get_system_prompt()

        assert "GENERATE_IMAGE" in prompt
        assert "prompt:" in prompt
        assert "negative_prompt:" in prompt
        assert "resolution:" in prompt
        assert "seed:" in prompt


class TestMediaAgentGenerate:
    """Tests for direct generation."""

    def test_generate_direct(self, temp_dir):
        """Test direct generation bypassing LLM."""
        agent = MediaAgent(output_dir=str(temp_dir))

        # Mock the generator
        mock_image = MagicMock()
        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_image
        agent._generator = mock_generator

        result = agent.generate(
            prompt="Test prompt",
            negative_prompt="blurry",
            resolution="1024x1024",
            seed=42,
        )

        mock_generator.generate.assert_called_once_with(
            prompt="Test prompt",
            negative_prompt="blurry",
            resolution="1024x1024",
            seed=42,
        )
        mock_image.save.assert_called_once()
        assert "generated_" in result
        assert ".png" in result


class TestMediaAgentUnload:
    """Tests for model unloading."""

    def test_unload_clears_models(self):
        """Test that unload clears all models."""
        agent = MediaAgent()

        # Set mock models
        mock_generator = MagicMock()
        mock_llm = MagicMock()
        agent._generator = mock_generator
        agent._llm = mock_llm

        agent.unload()

        mock_generator.unload.assert_called_once()
        mock_llm.unload.assert_called_once()
        assert agent._generator is None
        assert agent._llm is None

    def test_unload_handles_none(self):
        """Test that unload handles None models gracefully."""
        agent = MediaAgent()

        # Should not raise
        agent.unload()

        assert agent._generator is None
        assert agent._llm is None


class TestMediaAgentContextManager:
    """Tests for context manager."""

    def test_context_manager(self):
        """Test agent works as context manager."""
        with patch.object(MediaAgent, 'unload') as mock_unload:
            with MediaAgent() as agent:
                pass

            mock_unload.assert_called_once()

    def test_context_manager_on_exception(self):
        """Test cleanup happens even on exception."""
        with patch.object(MediaAgent, 'unload') as mock_unload:
            try:
                with MediaAgent() as agent:
                    raise ValueError("Test error")
            except ValueError:
                pass

            mock_unload.assert_called_once()
