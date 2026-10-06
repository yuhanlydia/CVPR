#!/usr/bin/env bash
set -euo pipefail

: "${GPU_HOST:?Set GPU_HOST=user@hostname}"
: "${GPU_PROJECT_DIR:?Set GPU_PROJECT_DIR=/absolute/path/to/CVPR}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: GPU_HOST=... GPU_PROJECT_DIR=... bash scripts/run_remote.sh '<command>'" >&2
  exit 2
fi

REMOTE_CMD="$1"
ssh "$GPU_HOST" "cd '$GPU_PROJECT_DIR' && git rev-parse HEAD && $REMOTE_CMD"
