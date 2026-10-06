#!/usr/bin/env bash
set -euo pipefail

: "${WAN_ROOT:?Set WAN_ROOT}"
: "${WAN_CKPT:?Set WAN_CKPT}"
: "${VBENCH_ROOT:?Set VBENCH_ROOT}"

OUT=${ROUND_OUT:-artifacts/round_002}
mkdir -p "$OUT"
python3 scripts/inspect_host.py --output "$OUT/host.json"

ARGS=(
  --wan-root "$WAN_ROOT"
  --ckpt-dir "$WAN_CKPT"
  --vbench-json "$VBENCH_ROOT/vbench/VBench_full_info.json"
  --output-dir "$OUT"
  --num-prompts "${NUM_PROMPTS:-3}"
  --sampling-steps "${SAMPLING_STEPS:-50}"
  --frame-num "${FRAME_NUM:-81}"
  --width "${WIDTH:-832}"
  --height "${HEIGHT:-480}"
  --shift "${SHIFT:-8.0}"
  --guide-scale "${GUIDE_SCALE:-6.0}"
  --base-seed "${BASE_SEED:-20261006}"
  --max-wall-hours "${MAX_WALL_HOURS:-7.5}"
)

if [[ -n "${FORCE_STEPS:-}" ]]; then
  ARGS+=(--force-steps "$FORCE_STEPS")
fi

set +e
python3 experiments/wan_cache/probe_cache_propagation.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
STATUS=${PIPESTATUS[0]}
set -e
COMMIT=$(git rev-parse HEAD)
python3 experiments/wan_cache/make_result_packet.py --round-dir "$OUT" --commit "$COMMIT" --exit-code "$STATUS"
exit "$STATUS"
