# Dot-source this from the repo root to prepare a shell for this lab:
#   . .\scripts\env.ps1
#
# Why these are needed on this machine:
#   USE_TF / USE_FLAX  - the venv is created with --system-site-packages, so it
#                        sees conda dl_env's TensorFlow (built against NumPy 1.x)
#                        while NumPy is 2.x. transformers lazily imports TF from
#                        image_transforms.py, which crashes the peft/trl import
#                        chain. Setting these to 0 keeps transformers away from TF.
#   HF_ENDPOINT        - HF mirror; useful on CN networks. Set only if unset.
#   HF_HUB_OFFLINE     - set to 1 when the dataset/model is already cached but the
#                        run still stalls on a huggingface.co HEAD request
#                        (observed with yahma/alpaca-cleaned). Pass -Offline to
#                        force it; omit to allow downloads.
#
# Usage:
#   . .\scripts\env.ps1            # mirror, online allowed
#   . .\scripts\env.ps1 -Offline   # use only the local HF cache

param([switch]$Offline)

$env:USE_TF = "0"
$env:USE_FLAX = "0"
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if ($Offline) { $env:HF_HUB_OFFLINE = "1" }

$repoRoot = Split-Path -Parent $PSScriptRoot
Write-Host "[env] USE_TF=0 USE_FLAX=0 HF_ENDPOINT=$env:HF_ENDPOINT" -ForegroundColor Cyan
if ($Offline) { Write-Host "[env] HF_HUB_OFFLINE=1 (local cache only)" -ForegroundColor Cyan }
Write-Host "[env] repo: $repoRoot" -ForegroundColor Cyan
Write-Host "[env] run:  .\.venv\Scripts\python.exe scripts\train_sft.py --config configs\sft_qwen05b_8g.yaml" -ForegroundColor Cyan

