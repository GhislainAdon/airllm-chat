# AirLLM Chat - Release Notes Template
# Used by .github/workflows/release.yml

## What's New

- 🪟 **Windows binary** — Standalone `.exe`, no Python needed
- 🐧 **Linux binary** — Pre-built for Ubuntu/Debian/Fedora
- 🍎 **macOS binary** — Universal build for Intel & Apple Silicon
- 📦 **Source distribution** — tar.gz and zip archives

## Downloads

| Platform | File | Description |
|----------|------|-------------|
| 🪟 Windows | `airllm-chat-windows-x64.zip` | Standalone .exe (Python bundled) |
| 🐧 Linux | `airllm-chat-linux-x64.tar.gz` | Pre-built binary (Python bundled) |
| 🍎 macOS | `airllm-chat-macos.zip` | Pre-built binary (Python bundled) |
| 📦 Source | `airllm-chat-src.tar.gz` | Source code (requires Python) |
| 📦 Source | `airllm-chat-src.zip` | Source code (requires Python) |

## Quick Start

### Windows
1. Download `airllm-chat-windows-x64.zip`
2. Extract all files
3. Double-click `airllm-chat.exe`
4. Browser opens at http://127.0.0.1:7860

### Linux
```bash
tar -xzf airllm-chat-linux-x64.tar.gz
cd airllm-chat
chmod +x airllm-chat
./airllm-chat
```

### macOS
```bash
unzip airllm-chat-macos.zip
cd airllm-chat
chmod +x airllm-chat
./airllm-chat
```

### From Source
```bash
tar -xzf airllm-chat-src.tar.gz
cd airllm-chat
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

## Use with VS Code Extensions (Cline, Continue, etc.)

Start the binary, then configure your extension:
- **Base URL**: `http://127.0.0.1:7860/v1`
- **API Key**: `sk-local` (or anything)
- **Model**: Select from the UI or set `--model` flag

## Verify Downloads

All files have SHA256 checksums. Verify with:
```bash
sha256sum -c SHA256SUMS.txt
```
