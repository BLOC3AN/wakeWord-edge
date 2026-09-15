#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements-train.txt"
"$ROOT/.venv/bin/pip" install -e "$ROOT/models/micro-wake-word"
echo "Environment ready: $ROOT/.venv"
