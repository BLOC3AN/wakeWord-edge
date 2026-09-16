#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
mkdir -p outputs
exec >>outputs/overnight.log 2>&1

if [[ "${1:-}" != "--train-only" ]]; then
  .venv/bin/python -u pipeline/prepare_speech_commands.py \
    data/external/base/downloads/speech_commands_v0.02 \
    data/external/base
fi
CUDA_VISIBLE_DEVICES=2,3 TF_CUDNN_USE_AUTOTUNE=0 .venv/bin/python -u pipeline/pretrain_embedding.py configs/base_pretrain.yaml
