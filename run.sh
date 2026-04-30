#!/usr/bin/env bash
# ===========================================
#  AirLLM Chat - Run Script for Linux/macOS
#  Runs directly from Python source
# ===========================================

set -e

echo ""
echo "  AirLLM Chat - Starting..."
echo ""

# Activate venv if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[Warning] No virtual environment found. Using system Python."
fi

# Check airllm
if ! python3 -c "import airllm" 2>/dev/null; then
    echo "[Installing] airllm not found. Installing dependencies..."
    pip install airllm
fi

# Run
python3 app.py "$@"
