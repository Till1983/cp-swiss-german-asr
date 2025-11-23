#!/bin/bash
# Setup script for RunPod environment: installs Python dependencies and downloads KenLM ARPA model

set -e

echo "🔎 [setup_runpod.sh] Starting setup for RunPod..."

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# 1. Install Python dependencies
echo "📦 Installing Python dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 2. Download KenLM ARPA file using download_lm.py
echo "⬇️  Downloading KenLM ARPA model from HuggingFace..."
python3 scripts/download_lm.py

echo "✅ Setup complete! KenLM ARPA file should be in src/models/lm/kenLM.arpa"