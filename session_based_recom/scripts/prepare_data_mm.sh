#!/usr/bin/env bash
# Build the MULTIMODAL TSTR datasets for DIMO / MMSBR from the same store-pairing
# config as prepare_data.sh (same splits, same shared item vocabulary -- run
# prepare_data.sh first if you also want the id-only dirs).
#
# Produces four sibling dirs under session_based_recom/data_processed/:
#   {real,synth}_{dimo,mmsbr}_paper/
#
# Feature extraction (BERT / GoogLeNet / CLIP) is cached per product under
# data_processed/mm_cache/ -- re-runs only redo the cheap dataset emission.
#
# Usage:
#   scripts/prepare_data_mm.sh [config] [extra preprocess_mm.py flags...]
#   PYTHON=/path/to/python scripts/prepare_data_mm.sh [...]  # explicit env
#
# Examples:
#   scripts/prepare_data_mm.sh                                   # everything
#   scripts/prepare_data_mm.sh conf/tstr_data.yaml --methods dimo
#   scripts/prepare_data_mm.sh conf/tstr_data.yaml --sample 20000 --pseudo mirror
set -euo pipefail

SBR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SBR/.." && pwd)"
CONFIG="${1:-$SBR/conf/tstr_data.yaml}"
PYTHON_BIN="${PYTHON:-python3}"
if [ $# -gt 0 ]; then shift; fi

# make the config path absolute so it survives the cd into REPO_ROOT
case "$CONFIG" in /*) : ;; *) CONFIG="$SBR/$CONFIG" ;; esac

# Fail before the expensive build with an actionable message. Feature-cache hits
# still need these packages for dataset emission and PCA.
if ! "$PYTHON_BIN" -c 'import numpy, scipy, sklearn, yaml, torch' 2>/dev/null; then
  echo "ERROR: $PYTHON_BIN does not have the multimodal preprocessing dependencies." >&2
  echo "Activate the project environment or install session_based_recom/requirements.txt." >&2
  echo "You can also select it explicitly: PYTHON=/path/to/python scripts/prepare_data_mm.sh" >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys; print(sys.executable)'))"
echo "Output root: $SBR/data_processed"

# data_root defaults to CWD; run from the outer repo root so data/ and results_*/ resolve.
( cd "$REPO_ROOT" && "$PYTHON_BIN" "$SBR/preprocess/preprocess_mm.py" --config "$CONFIG" "$@" )

echo
echo "Generated multimodal TSTR directories:"
for out_dir in \
  "$SBR/data_processed/real_dimo_paper" \
  "$SBR/data_processed/synth_dimo_paper" \
  "$SBR/data_processed/real_mmsbr_paper" \
  "$SBR/data_processed/synth_mmsbr_paper"; do
  if [ -d "$out_dir" ]; then
    echo "  $out_dir"
  fi
done
echo "Run the TSTR comparison with:  METHODS=\"dimo mmsbr\" scripts/tstr.sh"
