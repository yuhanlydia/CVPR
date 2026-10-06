#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${WAN_ROOT:?Set WAN_ROOT}"
: "${WAN_CKPT:?Set WAN_CKPT}"
: "${VBENCH_ROOT:?Set VBENCH_ROOT}"
PYTHON_BIN=${WAN_PYTHON:-python3}
exec "$PYTHON_BIN" tools/run_wan_window.py --hours "${MAX_WALL_HOURS:-2}" --execute
