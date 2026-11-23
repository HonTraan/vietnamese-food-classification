#!/usr/bin/env bash
# Local tests for sync_to_drive.sh / pull_results.sh (no Python, no Drive needed).
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SYNC="$HERE/sync_to_drive.sh"
PULL="$HERE/pull_results.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1"; exit 1; }

setup_drive() {
  rm -rf "$TMP/vnfood_colab" "$TMP/results"
  mkdir -p "$TMP/vnfood_colab/runs/E1_cnn"
  echo '{"test_accuracy": 0.6}' > "$TMP/vnfood_colab/runs/E1_cnn/metrics.json"
  echo "weights" > "$TMP/vnfood_colab/runs/E1_cnn/best.pt"
}

test_sync_writes_version() {
  setup_drive
  VNFOOD_DRIVE_ROOT="$TMP/vnfood_colab" bash "$SYNC" > "$TMP/out.txt" || fail "sync exited non-zero"
  local code="$TMP/vnfood_colab/code"
  for key in version commit dirty synced_at; do
    grep -q "^$key=" "$code/VERSION.txt" || fail "VERSION.txt missing $key"
  done
  grep -q "^version=" "$TMP/out.txt" || fail "sync did not print version"
  [[ -f "$code/src/train.py" ]] || fail "src/train.py not synced"
  [[ -f "$code/colab/sync_to_drive.sh" ]] || fail "colab/ not synced"
  for absent in .git data models colab/results "Deep Learning_Project_HK-2631.pdf"; do
    [[ ! -e "$code/$absent" ]] || fail "$absent should not be synced"
  done
}

test_sync_keeps_runs() {
  setup_drive
  VNFOOD_DRIVE_ROOT="$TMP/vnfood_colab" bash "$SYNC" > /dev/null || fail "sync exited non-zero"
  [[ -f "$TMP/vnfood_colab/runs/E1_cnn/metrics.json" ]] || fail "runs/ was touched by sync"
}

test_sync_refuses_wrong_target() {
  rm -rf "$TMP/other"
  if VNFOOD_DRIVE_ROOT="$TMP/other" bash "$SYNC" > /dev/null 2>&1; then
    fail "sync accepted a root not named vnfood_colab"
  fi
  [[ ! -e "$TMP/other/code" ]] || fail "sync wrote into wrong target"
}

test_pull_skips_pt() {
  setup_drive
  VNFOOD_DRIVE_ROOT="$TMP/vnfood_colab" VNFOOD_RESULTS_DIR="$TMP/results" bash "$PULL" > /dev/null \
    || fail "pull exited non-zero"
  [[ -f "$TMP/results/E1_cnn/metrics.json" ]] || fail "metrics.json not pulled"
  [[ ! -e "$TMP/results/E1_cnn/best.pt" ]] || fail "best.pt should not be pulled"
}

passed=0
for t in test_sync_writes_version test_sync_keeps_runs test_sync_refuses_wrong_target test_pull_skips_pt; do
  "$t"
  passed=$((passed + 1))
done
echo "$passed passed"
