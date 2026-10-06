#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--inside-budget" ]]; then
    SETUP_SECONDS=$(python3 -c 'import sys; sys.path.insert(0,"tools"); from window_budget import ensure_window,remaining_seconds; window=ensure_window("."); seconds=int(min(3600,remaining_seconds(window))); assert seconds>=120,"Retained window budget exhausted"; print(seconds)')
    exec timeout --signal=TERM --kill-after=10s "${SETUP_SECONDS}s" bash tools/bootstrap.sh --inside-budget
fi
python3 -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ is required"'
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --disable-pip-version-check torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
mkdir -p sources
if [[ ! -d sources/Qwen3-VL-Embedding ]]; then
    git clone --no-checkout https://github.com/QwenLM/Qwen3-VL-Embedding.git sources/Qwen3-VL-Embedding
    git -C sources/Qwen3-VL-Embedding checkout --detach 393e2978d27852b0d0230d6994f37f9c15bed73c
fi
.venv/bin/python tools/host_check.py --finish-setup
