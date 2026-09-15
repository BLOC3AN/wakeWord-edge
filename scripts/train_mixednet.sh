#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:?usage: $0 configs/mixednet.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/models/micro-wake-word:${PYTHONPATH:-}"
exec "$PYTHON" "$ROOT/models/micro-wake-word/microwakeword/model_train_eval.py" --training_config "$ROOT/$CONFIG" mixednet
