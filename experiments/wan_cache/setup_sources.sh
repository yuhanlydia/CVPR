#!/usr/bin/env bash
set -euo pipefail

ROOT="${EXTERNAL_ROOT:-external}"
mkdir -p "$ROOT"

WAN_SHA="9737cba9c1c3c4d04b33fcad41c111989865d315"
TEACACHE_SHA="7c10efc4702c6b619f47805f7abe4a7a08085aa0"
VBENCH_SHA="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
HF_MODEL_REV="37ec512624d61f7aa208f7ea8140a131f93afc9a"

clone_pin() {
  local url="$1"
  local dir="$2"
  local sha="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$url" "$dir"
  fi
  git -C "$dir" fetch origin "$sha" || git -C "$dir" fetch origin
  git -C "$dir" checkout --detach "$sha"
}

clone_pin https://github.com/Wan-Video/Wan2.1.git "$ROOT/Wan2.1" "$WAN_SHA"
clone_pin https://github.com/ali-vilab/TeaCache.git "$ROOT/TeaCache" "$TEACACHE_SHA"
clone_pin https://github.com/Vchitect/VBench.git "$ROOT/VBench" "$VBENCH_SHA"

if [[ "${INSTALL_WAN_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -r "$ROOT/Wan2.1/requirements.txt"
fi

MODEL_DIR="${WAN_CKPT:-$ROOT/Wan2.1-T2V-1.3B}"
if [[ "${DOWNLOAD_WAN_MODEL:-0}" == "1" ]]; then
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    python3 -m pip install "huggingface_hub[cli]"
  fi
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --revision "$HF_MODEL_REV" --local-dir "$MODEL_DIR"
fi

echo "WAN_ROOT=$ROOT/Wan2.1"
echo "TEACACHE_ROOT=$ROOT/TeaCache"
echo "VBENCH_ROOT=$ROOT/VBench"
echo "WAN_CKPT=$MODEL_DIR"
echo "WAN_MODEL_REV=$HF_MODEL_REV"
