#!/usr/bin/env bash
set -euo pipefail

: "${GPU_HOST:?Set GPU_HOST=user@hostname}"
: "${GPU_PROJECT_DIR:?Set GPU_PROJECT_DIR=/absolute/path/to/CVPR}"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: GPU_HOST=... GPU_PROJECT_DIR=... bash scripts/run_remote.sh '<command>'" >&2
  exit 2
fi

REMOTE_CMD="$1"
QUOTED_PROJECT_DIR=$(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$GPU_PROJECT_DIR")
ssh -- "$GPU_HOST" "cd -- $QUOTED_PROJECT_DIR && git rev-parse HEAD && $REMOTE_CMD"
