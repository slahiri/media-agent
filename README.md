# MediaAgent

[![PyPI version](https://img.shields.io/pypi/v/media-agent.svg)](https://pypi.org/project/media-agent/)
[![Python](https://img.shields.io/pypi/pyversions/media-agent.svg)](https://pypi.org/project/media-agent/)
[![License](https://img.shields.io/github/license/slahiri/media-agent.svg)](https://github.com/slahiri/media-agent/blob/main/LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/slahiri/media-agent/publish.yml?branch=main)](https://github.com/slahiri/media-agent/actions)
[![Downloads](https://img.shields.io/pypi/dm/media-agent.svg)](https://pypi.org/project/media-agent/)

AI-powered image generation with natural language. Features an interactive CLI with multiple LLM providers and image backends.

## Quick Start

```bash
# Install
pip install -e .

# Start the CLI
media-agent
```

## Setup

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate

# Basic install
pip install -e .

# With cloud LLM providers (recommended)
pip install -e ".[all-llm]"
```

### 2. Configure an LLM Provider

Choose one:

**OpenAI** (easiest)
```
/settings openai.api_key sk-your-key-here
/llm openai
```

**Anthropic**
```
/settings anthropic.api_key sk-ant-your-key-here
/llm anthropic
```

**Ollama** (free, local)
```bash
# Install Ollama first: https://ollama.ai
ollama pull llama3.2
ollama serve
```
```
/llm ollama llama3.2
```

**Local HuggingFace** (requires GPU)
```bash
media-agent download llm
```
```
/llm huggingface
```

### 3. Configure Image Generation

**Local GPU** (default, requires CUDA GPU)
```bash
media-agent download image
```
```
/image local
```

**Nanobanana Cloud** (no GPU needed)
```
/settings nanobanana.api_key your-key-here
/image nanobanana
```

### 4. Generate Images

Just chat naturally:
```
> Generate a sunset over mountains
> Create a cyberpunk city at night
> A cat wearing a tiny hat
```

Or use the direct command:
```
/generate a beautiful forest landscape
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/generate <prompt>` | Generate an image |
| `/llm <provider>` | Switch LLM (openai, anthropic, ollama, huggingface) |
| `/image <provider>` | Switch image backend (local, nanobanana) |
| `/settings <key> <value>` | Configure settings |
| `/resolutions` | Show available image sizes |
| `/unload` | Free GPU memory |
| `/clear` | Clear chat history |
| `/quit` | Exit |

## Command Line Usage

```bash
# Interactive mode
media-agent

# Generate directly
media-agent generate "a sunset over mountains"

# With options
media-agent generate "a cat" -r 1024x1024 -s 42

# Download models
media-agent download all

# Manage settings
media-agent settings list
media-agent settings openai.api_key sk-xxx
```

## Python API

```python
from media_agent import MediaAgent

# Using configured settings
agent = MediaAgent()
result = agent.run("Generate a sunset over mountains")
print(result)

# Specify providers
agent = MediaAgent(
    llm_provider="openai",
    llm_model="gpt-4o",
    image_provider="local",
)

# Direct generation (skip LLM)
path = agent.generate(
    prompt="A serene Japanese garden",
    resolution="1344x768",
    seed=42,
)

# Cleanup
agent.unload()
```

## Image Resolutions

| Name | Size | Best For |
|------|------|----------|
| `1024x1024` | Square | Balanced compositions |
| `1344x768` | 16:9 | Landscapes, wide scenes |
| `768x1344` | 9:16 | Portraits, tall subjects |
| `1152x896` | 4:3 | Slight landscape |
| `896x1152` | 3:4 | Slight portrait |

## Settings Location

Settings are stored in `~/.config/media_agent/settings.json`. No environment variables needed.

## Requirements

- Python >= 3.10
- For local image generation: CUDA GPU with 8GB+ VRAM
- For cloud providers: API key only

## License

MIT
