# AirLLM Chat

[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/airllm-chat/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/YOUR_USERNAME/airllm-chat/actions)
[![Release](https://img.shields.io/github/v/release/YOUR_USERNAME/airllm-chat?style=flat-square&color=green)](https://github.com/YOUR_USERNAME/airllm-chat/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)](#)

A ChatGPT-style local AI chat interface powered by the **airllm** library. Run large language models entirely on your machine — no cloud, no API keys, no data sent externally.

**Pre-built binaries available on [GitHub Releases](https://github.com/YOUR_USERNAME/airllm-chat/releases) — no Python needed!**

## Features

- **ChatGPT-style UI** — Dark theme, streaming responses, chat history, markdown formatting
- **OpenAI-compatible API** — Works with Cline, Continue, Cursor, OpenInterpreter, and any OpenAI client
- **Local inference** — All processing runs on your machine via airllm
- **Model management** — Load/unload models from the sidebar, pick from popular options
- **Streaming** — See tokens generated in real-time with Server-Sent Events
- **Persistent history** — Conversations saved to localStorage
- **No dependencies to install for end users** — Ship as a single `.exe` on Windows

## Quick Start

### Option 1: Run from Source (Windows)

```bash
# 1. Clone or download this folder
cd airllm-chat

# 2. Create virtual environment and install
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the app
python app.py

# Or use the convenient script:
run.bat
```

Then open **http://127.0.0.1:7860** in your browser.

### Option 2: Build a Windows `.exe` Binary

```bash
# Double-click or run:
build.bat

# The executable will be at:
#   dist\airllm-chat\airllm-chat.exe
```

Copy the entire `dist\airllm-chat\` folder to any Windows machine and run `airllm-chat.exe`.

### Option 3: Run on Linux/macOS

```bash
cd airllm-chat
python3 -m venv venv
source venv/bin/activate
pip install airllm
python app.py
```

## Use with Cline / Continue / Cursor (VS Code Extensions)

This is exactly how Ollama works. AirLLM Chat exposes an **OpenAI-compatible API**, so any client that supports "OpenAI Compatible" or "custom OpenAI endpoint" will work.

### Setup (same for all clients)

1. Start the server:
```bash
python app.py --model garage-bAInd/Platypus2-70B-instruct --no-browser
```

2. In your VS Code extension, configure:
   - **API Provider**: OpenAI Compatible (or Ollama-compatible)
   - **Base URL**: `http://127.0.0.1:7860/v1`
   - **API Key**: anything (e.g. `sk-local`, it's ignored)
   - **Model**: the model name you loaded (or `no-model-loaded` to auto-switch)

### Cline
```
API Provider: OpenAI Compatible
Base URL: http://127.0.0.1:7860/v1
Model ID: garage-bAInd/Platypus2-70B-instruct
API Key: sk-local
```

### Continue
```json
{
  "models": [{
    "title": "AirLLM Local",
    "provider": "ollama",
    "model": "garage-bAInd/Platypus2-70B-instruct",
    "apiBase": "http://127.0.0.1:7860/v1"
  }]
}
```

### Open Interpreter
```bash
interpreter --api_base http://127.0.0.1:7860/v1 --model garage-bAInd/Platypus2-70B-instruct
```

### OpenAI Python SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:7860/v1", api_key="sk-local")
resp = client.chat.completions.create(
    model="garage-bAInd/Platypus2-70B-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(resp.choices[0].message.content)
```

### curl
```bash
curl http://127.0.0.1:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"garage-bAInd/Platypus2-70B-instruct","messages":[{"role":"user","content":"Hello"}]}'
```

## Web UI Usage

1. **Load a model** — Select a model from the sidebar dropdown and click "Load Model"
2. **Chat** — Type your message and press Enter (Shift+Enter for new line)
3. **Adjust parameters** — Tweak temperature and max tokens in the sidebar
4. **Manage conversations** — Start new chats, switch between them in the sidebar

### Command Line Options

```
python app.py [OPTIONS]

Options:
  --port PORT          Port to run the server on (default: 7860)
  --host HOST          Host to bind to (default: 127.0.0.1)
  --model MODEL        Model path or HuggingFace repo ID to auto-load
  --max-tokens N       Max new tokens for generation (default: 512)
  --temperature T      Temperature for generation (default: 0.7)
  --no-browser         Don't open browser automatically
```

### Examples

```bash
# Auto-load a model on startup
python app.py --model garage-bAInd/Platypus2-70B-instruct

# Use a local GGUF model file
python app.py --model "C:\Models\llama-7b-chat.Q4_K_M.gguf"

# Expose on local network
python app.py --host 0.0.0.0 --port 8080
```

## CI/CD & Releases

The project includes a full GitHub Actions pipeline:

### Pipelines

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| **CI** | Push / PR to main | Lint, type check, test, build smoke test |
| **Build** | Push to main / Manual | Build binaries for all platforms (artifacts, no release) |
| **Release** | Tag push (`v*`) | Build all platforms + create GitHub Release |

### How to Create a Release

```bash
# 1. Tag your version
git tag v1.0.0
git push origin v1.0.0

# 2. GitHub Actions automatically:
#    - Builds Windows .exe
#    - Builds Linux binary (tar.gz)
#    - Builds macOS binary (zip)
#    - Creates source tarball + zip
#    - Generates SHA256 checksums
#    - Publishes everything as a GitHub Release
```

### Manual Build (without release)

Go to **Actions > Build** in GitHub, click **Run workflow**, select which platforms to build. Binaries are available as artifacts for 14 days.

### Release Artifacts

| File | Platform | Description |
|------|----------|-------------|
| `airllm-chat-windows-x64.zip` | Windows | Standalone .exe bundle |
| `airllm-chat-linux-x64.tar.gz` | Linux | Pre-built binary |
| `airllm-chat-macos.zip` | macOS | Pre-built binary |
| `airllm-chat-src.tar.gz` | All | Source code |
| `airllm-chat-src.zip` | All | Source code |
| `SHA256SUMS.txt` | All | Integrity checksums |

## Supported Models

airllm supports various quantized model formats. Popular choices:

| Model | Size | Format |
|-------|------|--------|
| garage-bAInd/Platypus2-70B-instruct | 70B | GPTQ |
| garage-bAInd/Platypus2-13B-instruct | 13B | GPTQ |
| TheBloke/Llama-2-7B-Chat-GPTQ | 7B | GPTQ |
| Any GGUF model | Varies | GGUF |

You can also use **any local model path** — just type or paste the path in the model selection dropdown (editable).

## Project Structure

```
airllm-chat/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml          # CI pipeline (lint, test, build check)
│   │   ├── build.yml       # On-demand build (manual trigger)
│   │   └── release.yml     # Full release (tag trigger)
│   └── RELEASE_NOTES.md    # Release notes template
├── app.py                  # Main server & API endpoints (OpenAI-compatible)
├── airllm_engine.py        # airllm wrapper engine
├── airllm-chat.spec        # PyInstaller build spec
├── build.bat               # Windows build script
├── build.sh                # Linux/macOS build script
├── run.bat                 # Windows run script
├── run.sh                  # Linux/macOS run script
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Main HTML page
├── static/
│   ├── style.css           # Dark theme styles
│   └── app.js              # Frontend JavaScript
├── LICENSE                 # MIT License
└── README.md               # This file
```

## System Requirements

- **Python 3.9+**
- **CUDA-capable GPU recommended** (CPU works but will be very slow)
- **RAM**: 8GB minimum, 32GB+ recommended for 70B models
- **Disk**: 20GB+ for model weights
- **OS**: Windows 10/11, Linux, macOS

## Troubleshooting

**"No model loaded"** — Select and load a model from the sidebar before chatting.

**Model download is slow** — First-time downloads from HuggingFace can be large. Use a local model path if available.

**Out of memory** — Try a smaller model (7B instead of 70B), or reduce batch sizes.

**Build fails with PyInstaller** — Make sure all dependencies are installed in the virtual environment. Try deleting `venv/` and `build/` folders, then rebuild.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | ChatGPT-style web UI |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming supported) |
| `/v1/models` | GET | List available models (OpenAI format) |
| `/api/chat` | POST | Internal chat endpoint |
| `/api/load` | POST | Load a model |
| `/api/settings` | POST | Update temperature/tokens |
| `/api/info` | GET | Engine status |

## License

[MIT](LICENSE)
