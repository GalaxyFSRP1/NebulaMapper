@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_TELEMETRY=1
title NebulaMapper - PhantomVoid Neural BeatStudio
echo ========================================================
echo   NebulaMapper - PhantomVoid Neural BeatStudio
echo   GitHub: https://github.com/GalaxyFSRP1/NebulaMapper
echo   Models: https://huggingface.co/PhantomVoidTTV/NebulaMapper
echo ========================================================
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)
python -c "import bsgen, gradio, safetensors, huggingface_hub, yt_dlp, librosa" >nul 2>&1
if errorlevel 1 (
    echo [1/3] First-time setup: Installing Python dependencies...
    pip install -e .
) else (
    echo [1/3] Dependencies already installed ^(fast startup^).
)
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [2/3] NVIDIA GPU detected with CPU-only PyTorch! Installing CUDA GPU PyTorch...
        pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall --no-deps
    ) else (
        echo [2/3] NVIDIA CUDA GPU acceleration is active.
    )
) else (
    echo [2/3] Running in CPU Turbo Mode.
)
echo.
echo [3/3] Starting NebulaMapper Studio on http://127.0.0.1:7860 ...
python -X utf8 scripts/webui.py --host 127.0.0.1 --port 7860
pause
