#!/usr/bin/env bash
# generated_unexecuted. Foreground Local CPU setup under the single run_harness.
# Existing native prefix is created separately; no model download or GPU probing.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash tools/setup_native_env.sh ACTUAL_CONDA_PREFIX NEW_RECEIPT_DIR\n' >&2
    exit 2
fi
CVPR_SETUP_PREFIX=$1
CVPR_SETUP_RECEIPTS=$2
CVPR_SETUP_PROJECT=$(cd "$(dirname "$0")/.." && pwd)
[[ "$CVPR_SETUP_PREFIX" = /* && -d "$CVPR_SETUP_PREFIX/conda-meta" ]] || {
    printf 'Actual native Conda prefix required; create it as an admitted setup job first.\n' >&2
    exit 2
}
[[ "$CVPR_SETUP_RECEIPTS" = /* && ! -e "$CVPR_SETUP_RECEIPTS" ]] || {
    printf 'A new absolute receipt directory is required; retained receipts are never overwritten.\n' >&2
    exit 2
}
mkdir -p "$CVPR_SETUP_RECEIPTS"
cd "$CVPR_SETUP_PROJECT"
trap 'CVPR_SETUP_EXIT=$?; printf "%s\n" "$CVPR_SETUP_EXIT" > "$CVPR_SETUP_RECEIPTS/exit-code.txt"; date -u +%FT%TZ > "$CVPR_SETUP_RECEIPTS/finished-at.txt"' EXIT
date -u +%FT%TZ > "$CVPR_SETUP_RECEIPTS/started-at.txt"
git rev-parse HEAD > "$CVPR_SETUP_RECEIPTS/project-commit.txt"
sha256sum requirements.txt requirements-assets.txt configs/environment-native.yml \
    tools/setup_native_env.sh > "$CVPR_SETUP_RECEIPTS/source-sha256.txt"
conda run --no-capture-output --prefix "$CVPR_SETUP_PREFIX" python --version
conda run --no-capture-output --prefix "$CVPR_SETUP_PREFIX" python -m pip install \
    --disable-pip-version-check torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126 \
    --report "$CVPR_SETUP_RECEIPTS/torch-install.json"
conda run --no-capture-output --prefix "$CVPR_SETUP_PREFIX" python -m pip install \
    --disable-pip-version-check -r requirements.txt -r requirements-assets.txt \
    --report "$CVPR_SETUP_RECEIPTS/dependencies-install.json"
conda list --prefix "$CVPR_SETUP_PREFIX" --explicit > "$CVPR_SETUP_RECEIPTS/conda-explicit.txt"
conda run --no-capture-output --prefix "$CVPR_SETUP_PREFIX" python -m pip freeze --all \
    > "$CVPR_SETUP_RECEIPTS/pip-freeze.txt"
# Exit 0 means installation commands completed, not software/native/GPU acceptance.
