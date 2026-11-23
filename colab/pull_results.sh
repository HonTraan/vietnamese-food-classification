#!/usr/bin/env bash
# Copy experiment results from Google Drive back into colab/results/ (never *.pt checkpoints).
# Usage: bash colab/pull_results.sh [run_id]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${VNFOOD_DRIVE_ROOT:-}" ]]; then
  DRIVE_ROOT="$VNFOOD_DRIVE_ROOT"
else
  parents=("$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive")
  if [[ ${#parents[@]} -ne 1 || ! -d "${parents[0]}" ]]; then
    echo "Không tìm thấy đúng một thư mục Google Drive for Desktop; đặt VNFOOD_DRIVE_ROOT." >&2
    exit 1
  fi
  DRIVE_ROOT="${parents[0]}/DL/vnfood_colab"
fi
RESULTS_DIR="${VNFOOD_RESULTS_DIR:-$REPO/colab/results}"
RUNS="$DRIVE_ROOT/runs"
[[ -d "$RUNS" ]] || { echo "Chưa có $RUNS" >&2; exit 1; }

mkdir -p "$RESULTS_DIR"
if [[ -n "${1:-}" ]]; then
  rsync -a --exclude '*.pt' "$RUNS/$1" "$RESULTS_DIR/"
else
  rsync -a --exclude '*.pt' "$RUNS/" "$RESULTS_DIR/"
fi
echo "Pulled ${1:-all runs} into $RESULTS_DIR"
