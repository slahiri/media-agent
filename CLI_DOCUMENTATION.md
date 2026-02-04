# MediaAgent CLI Documentation

## Overview

MediaAgent is an AI-powered image generation CLI with:
- **Multi-provider LLM support** (OpenAI, Anthropic, Ollama, HuggingFace)
- **Multiple image backends** (Local Z-Image, Nanobanana API)
- **Rich interactive interface** using Rich + prompt_toolkit
- **Settings management via CLI** (no environment variables needed)

## Installation

```bash
# Basic install
pip install -e .

# With all LLM providers
pip install -e ".[all-llm]"

# With image preview support
pip install -e ".[preview]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```bash
# Start interactive CLI
media-agent

# Generate image directly
media-agent generate "a sunset over mountains"

# Download models
media-agent download all
```

## Architecture

```
media_agent/
├── cli/                          # CLI module
│   ├── __init__.py
│   ├── app.py                    # Main CLI controller
│   ├── commands.py               # Slash commands
│   ├── input_handler.py          # prompt_toolkit input
│   ├── renderer.py               # Rich message rendering
│   ├── progress.py               # Progress bars
│   ├── themes.py                 # Dark/light themes
│   └── components/
│       ├── header.py             # ASCII logo
│       ├── tips.py               # Getting started tips
│       └── help_overlay.py       # Help screen
├── llm/                          # Multi-provider LLM
│   ├── __init__.py
│   ├── base.py                   # BaseLLM abstract class
│   ├── settings.py               # Settings management (JSON file)
│   ├── registry.py               # LLM registry and factory
│   ├── openai_llm.py             # OpenAI provider
│   ├── anthropic_llm.py          # Anthropic provider
│   ├── ollama_llm.py             # Ollama provider
│   ├── huggingface_llm.py        # Local HuggingFace models
│   └── qwen.py                   # Legacy Qwen wrapper
├── image/
│   ├── generator.py              # Local Z-Image generator
│   └── nanobanana.py             # Nanobanana API client
├── tools/
│   ├── image_tool.py             # LangChain tool wrapper
│   └── downloader.py             # Model download utilities
├── agent/
│   └── agent.py                  # LangGraph MediaAgent
└── __main__.py                   # CLI entry point
```

## CLI Commands

### Chat & Navigation

| Command | Description |
|---------|-------------|
| `/help` | Show help screen |
| `/clear` | Clear chat history |
| `/history` | Show conversation history |
| `/quit`, `/exit` | Exit application |

### Image Generation

| Command | Description |
|---------|-------------|
| `/generate <prompt>` | Generate image directly |
| `/resolutions` | List available resolutions |
| `/unload` | Free GPU memory |

### LLM Provider

| Command | Description |
|---------|-------------|
| `/llm list` | List available providers |
| `/llm <provider> [model]` | Switch provider |
| `/llm openai gpt-4o` | Use OpenAI GPT-4o |
| `/llm anthropic claude-sonnet-4-20250514` | Use Claude |
| `/llm ollama llama3.2` | Use local Ollama |
| `/llm huggingface` | Use local HuggingFace model |

### Image Provider

| Command | Description |
|---------|-------------|
| `/image list` | List image providers |
| `/image local` | Use local Z-Image (GPU) |
| `/image local split` | Use split files mode |
| `/image nanobanana` | Use Nanobanana API |

### Settings

| Command | Description |
|---------|-------------|
| `/settings list` | Show all settings |
| `/settings <key> <value>` | Set a value |
| `/settings openai.api_key <key>` | Set OpenAI API key |
| `/settings anthropic.api_key <key>` | Set Anthropic key |
| `/settings nanobanana.api_key <key>` | Set Nanobanana key |
| `/settings ollama.base_url <url>` | Set Ollama URL |

### Interface

| Command | Description |
|---------|-------------|
| `/theme dark` | Switch to dark theme |
| `/theme light` | Switch to light theme |
| `?` | Quick help |
| `↑/↓` | Navigate history |
| `Tab` | Auto-complete |
| `Ctrl+C` | Cancel operation |

## Settings File

Settings are stored in `~/.config/media_agent/settings.json`:

```json
{
  "current_provider": "openai",
  "current_model": "gpt-4o",
  "openai": {
    "api_key": "sk-...",
    "default_model": "gpt-4o"
  },
  "anthropic": {
    "api_key": "sk-ant-...",
    "default_model": "claude-sonnet-4-20250514"
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.2"
  },
  "nanobanana": {
    "api_key": "nb-...",
    "base_url": "https://api.nanobanana.ai"
  },
  "image": {
    "provider": "local",
    "mode": "pipeline"
  }
}
```

## LLM Providers

### OpenAI
```bash
/settings openai.api_key sk-your-key-here
/llm openai gpt-4o
```

### Anthropic
```bash
/settings anthropic.api_key sk-ant-your-key-here
/llm anthropic claude-sonnet-4-20250514
```

### Ollama (Local)
```bash
# Start Ollama server first
ollama serve

/settings ollama.base_url http://localhost:11434
/llm ollama llama3.2
```

### HuggingFace (Local)
```bash
# Requires GPU and downloaded models
media-agent download llm
/llm huggingface
```

## Image Providers

### Local Z-Image (GPU Required)
```bash
# Download model
media-agent download image

# Use local generation
/image local
/image local pipeline  # Standard mode
/image local split     # Split files mode (ComfyUI-style)
```

### Nanobanana (Cloud API)
```bash
/settings nanobanana.api_key nb-your-key-here
/image nanobanana
```

## Image Resolutions

| Resolution | Ratio | Use Case |
|------------|-------|----------|
| 1024x1024 | 1:1 | Square, balanced |
| 1344x768 | 16:9 | Landscape, wide scenes |
| 768x1344 | 9:16 | Portrait, tall subjects |
| 1152x896 | 4:3 | Slight landscape |
| 896x1152 | 3:4 | Slight portrait |
| 1216x832 | 3:2 | Landscape |
| 832x1216 | 2:3 | Portrait |
| 1536x640 | 21:9 | Ultrawide, panoramas |
| 640x1536 | 9:21 | Very tall |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate command history |
| `Tab` | Auto-complete commands |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit application |
| `?` | Show quick help |

## Programmatic Usage

```python
from media_agent import MediaAgent

# Using default settings
agent = MediaAgent()
result = agent.run("Generate a sunset over mountains")

# Specify providers
agent = MediaAgent(
    llm_provider="openai",
    llm_model="gpt-4o",
    image_provider="nanobanana",
)

# Direct generation (bypass LLM)
path = agent.generate(
    prompt="A cat sitting on a beach",
    resolution="1024x1024",
)

# Cleanup
agent.unload()
```

## Themes

### Dark Theme (Default)
- Green user prefix
- Magenta assistant prefix
- Cyan system messages
- Dark background optimized

### Light Theme
- Blue user prefix
- Magenta assistant prefix
- Blue system messages
- Light background optimized

## Image Preview

When an image is generated:
1. Path is displayed as a clickable link (terminal dependent)
2. If `rich-pixels` is installed, inline ASCII preview is shown
3. Terminals with Kitty/iTerm2 protocol support show actual image

Install preview support:
```bash
pip install rich-pixels
```

## Troubleshooting

### "API key not set"
```bash
/settings <provider>.api_key <your-key>
```

### "Model not loaded"
```bash
# For local models
media-agent download all

# For Ollama
ollama pull llama3.2
```

### "CUDA out of memory"
```bash
/unload  # Free GPU memory
# Or use cloud provider
/image nanobanana
```

### "Connection refused" (Ollama)
```bash
# Start Ollama server
ollama serve

# Check URL
/settings ollama.base_url http://localhost:11434
```

## Version History

- **v0.3.0**: Multi-provider CLI with Rich interface
- **v0.2.0**: LangGraph agent with Z-Image integration
- **v0.1.0**: Initial release
