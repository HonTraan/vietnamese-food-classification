#!/usr/bin/env bash
# Create <repo>/.venv for running the notebooks on this Mac and register it as the Jupyter kernel "vnfood (.venv)".
# Usage: bash colab/setup_local.sh [--dry-run]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == no ]]; then "$@"; fi
}

DRY_RUN=no
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=yes

cd "$REPO"
run uv venv --python 3.12 .venv
run uv pip install --python .venv/bin/python -r requirements.txt ipykernel
run .venv/bin/python -m ipykernel install --user --name vnfood --display-name "vnfood (.venv)"
echo "Xong. Mở notebook và chọn kernel 'vnfood (.venv)'."
