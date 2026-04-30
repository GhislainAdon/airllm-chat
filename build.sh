#!/usr/bin/env bash
# ===========================================
#  AirLLM Chat - Build Script for Linux/macOS
#  Creates a standalone binary using PyInstaller
# ===========================================

set -e

echo ""
echo " ============================================"
echo "  AirLLM Chat - Build Script"
echo " ============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "Please install Python 3.9+ from python.org or your package manager."
    exit 1
fi

echo "[Step 1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "[Step 2/4] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install airllm
pip install pyinstaller

echo ""
echo "[Step 3/4] Building binary with PyInstaller..."
pyinstaller airllm-chat.spec --clean

echo ""
echo "[Step 4/4] Done!"
echo ""
echo " The binary is located at:"
echo "   dist/airllm-chat/airllm-chat"
echo ""
echo " To run it:"
echo "   cd dist/airllm-chat && ./airllm-chat"
echo ""
