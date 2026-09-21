@echo off
REM Run SFT baseline with env workarounds for this machine.
set USE_TF=0
set USE_FLAX=0
set HF_ENDPOINT=https://hf-mirror.com
cd /d "%~dp0.."
".venv\Scripts\python.exe" "scripts\train_sft.py" --config "configs\sft_qwen05b_8g.yaml"
