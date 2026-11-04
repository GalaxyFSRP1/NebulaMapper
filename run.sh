#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
echo "========================================================"
echo "  NebulaMapper — PhantomVoid Neural BeatStudio"
echo "  GitHub: https://github.com/GalaxyFSRP1/NebulaMapper"
echo "  Models: https://huggingface.co/PhantomVoidTTV/NebulaMapper"
echo "========================================================"
echo "[1/2] Installing / verifying Python dependencies..."
pip install -e .
echo "[2/2] Launching NebulaMapper Studio at http://127.0.0.1:7860 ..."
python3 -X utf8 scripts/webui.py --host 127.0.0.1 --port 7860
