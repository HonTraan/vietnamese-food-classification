#!/usr/bin/env bash
# Mirror the repo working tree into Google Drive (Drive for Desktop) so Colab can run it.
# Only ever writes to $VNFOOD_DRIVE_ROOT/code; data/ and runs/ on Drive are never touched.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

resolve_drive_root() {
  if [[ -n "${VNFOOD_DRIVE_ROOT:-}" ]]; then
    echo "$VNFOOD_DRIVE_ROOT"; return
  fi
  local parents=("$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive")
  if [[ ${#parents[@]} -ne 1 || ! -d "${parents[0]}" ]]; then
    echo "Không tìm thấy đúng một thư mục Google Drive for Desktop; đặt VNFOOD_DRIVE_ROOT." >&2
    exit 1
  fi
  echo "${parents[0]}/DL/vnfood_colab"
}

DRIVE_ROOT="$(resolve_drive_root)"
if [[ "$(basename "$DRIVE_ROOT")" != "vnfood_colab" ]]; then
  echo "Từ chối sync: thư mục đích phải tên vnfood_colab, nhận được: $DRIVE_ROOT" >&2
  exit 1
fi
CODE_DIR="$DRIVE_ROOT/code"
mkdir -p "$CODE_DIR"

rsync -a --delete \
  --exclude '.git/' --exclude '/data/' --exclude '/models/' --exclude '/colab/results/' \
  --exclude '__pycache__/' --exclude '.venv/' --exclude '.pytest_cache/' --exclude '.superpowers/' \
  --exclude 'graphify-out/' --exclude '*.pdf' --exclude '.DS_Store' --exclude '/VERSION.txt' \
  "$REPO/" "$CODE_DIR/"

commit="$(git -C "$REPO" rev-parse HEAD)"
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then dirty=yes; suffix="-dirty"; else dirty=no; suffix=""; fi
version="$(git -C "$REPO" rev-parse --short HEAD)${suffix}-$(date +%Y%m%d%H%M%S)"
{
  echo "version=$version"
  echo "commit=$commit"
  echo "dirty=$dirty"
  echo "synced_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "content_hash=$(cd "$REPO" && python3 -I -c 'import sys; sys.path.insert(0, "."); from colab.session import code_hash; print(code_hash("."))')"
} > "$CODE_DIR/VERSION.txt"

echo "Synced to $CODE_DIR"
echo "version=$version"
