# image-gen

A Python library for image generation using Z-Image-Turbo and LLM text generation using Qwen.

## Features

- **Image Generation**: Z-Image-Turbo (6B params) - photorealistic images with bilingual text rendering
- **LLM**: Qwen2.5-7B-Instruct for text generation and chat
- **Flexible Loading**: Support for HuggingFace pipeline, split files (ComfyUI-style), or local models
- **Config-driven**: YAML configuration for models, paths, and generation defaults
- **Model Downloader**: Utility to download and manage models

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Download Models

```bash
# Download all models (pipeline mode)
python -m image_gen.utils.downloader all

# Or download split files and copy to local models/ folder
python -m image_gen.utils.downloader all split --local
```

### 2. Generate Images

```python
from image_gen import ImageGenerator

# Using HuggingFace pipeline (auto-downloads)
gen = ImageGenerator(mode="pipeline")

# Or using local models
gen = ImageGenerator(mode="local")

# Generate an image
image = gen.generate(
    prompt="A serene mountain landscape at sunset",
    seed=42,
)
image.save("output.png")
```

### 3. Use LLM

```python
from image_gen import QwenLLM

llm = QwenLLM()

# Text generation
response = llm.generate("Explain quantum computing:")

# Chat format
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
])
```

## Models

### Z-Image-Turbo (Image Generation)

| Component | File | Size |
|-----------|------|------|
| Text Encoder | `qwen_3_4b.safetensors` | ~8GB |
| Diffusion Model | `z_image_turbo_bf16.safetensors` | ~12GB |
| VAE | `ae.safetensors` | ~335MB |

**Sources:**
- Pipeline: `Tongyi-MAI/Z-Image-Turbo`
- Split files: `Comfy-Org/z_image_turbo`

### Qwen LLM

- Default: `Qwen/Qwen2.5-7B-Instruct`
- Alternatives: `Qwen2.5-3B-Instruct` (smaller), `Qwen2.5-14B-Instruct` (larger)

## Configuration

Edit `config.yaml` to customize models and settings:

```yaml
models:
  image:
    pipeline:
      name: "Tongyi-MAI/Z-Image-Turbo"
    default_steps: 8
    default_size: [1024, 1024]
    guidance_scale: 1.0
  llm:
    name: "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: 512

paths:
  cache_dir: "~/.cache/image_gen"
  models_dir: "./models"
```

## Project Structure

```
image-generator/
├── config.yaml              # Model configurations
├── pyproject.toml           # Package dependencies
├── models/                  # Local models folder
│   ├── text_encoders/
│   ├── diffusion_models/
│   ├── vae/
│   └── llm/
├── image_gen/
│   ├── config.py            # Config loader
│   ├── image/
│   │   └── generator.py     # ImageGenerator class
│   ├── llm/
│   │   └── qwen.py          # QwenLLM class
│   └── utils/
│       └── downloader.py    # Model download utilities
└── examples/
    └── usage.py             # Example scripts
```

## CLI Commands

```bash
# List available models
python -m image_gen.utils.downloader list

# Download all models
python -m image_gen.utils.downloader all [pipeline|split] [--local]

# Download image model only
python -m image_gen.utils.downloader image [pipeline|split] [--local]

# Download LLM only
python -m image_gen.utils.downloader llm [--local]
```

## Requirements

- Python >= 3.10
- CUDA-capable GPU (16GB+ VRAM recommended)
- PyTorch >= 2.0

## License

MIT
