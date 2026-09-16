#!/usr/bin/env bash
# Build the Train-Synthetic-Test-Real (TSTR) datasets from a store-pairing
# config. No external method implementations are required or fetched.
#
# Produces three sibling dirs under session_based_recom/data_processed/:
#   real/   baseline  (train real  -> test real held-out)
#   synth/  full synthetic data retained for DIMO/MMSBR preprocessing
#   synth_matched/ synthetic examples volume-matched to real, used by
#                          NARM/RESTC/RAIN
# All share ONE item vocabulary, so a synth-trained model can score real sessions.
#
# The config's relative real/synth paths resolve against the CWD (data_root), so
# this script runs preprocess.py from the OUTER repo root (buyer-sim-gen/), where
# data/ and results_*/ live. The processed output still lands in
# session_based_recom/data_processed/ (an absolute path inside preprocess.py).
#
# Usage:
#   scripts/prepare_data.sh [config] [sample_rows] [match_seed]
#
# Examples:
#   scripts/prepare_data.sh                                  # full data
#   scripts/prepare_data.sh conf/tstr_data.yaml 20000        # smoke sample
set -euo pipefail

SBR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SBR/.." && pwd)"
CONFIG="${1:-$SBR/conf/tstr_data.yaml}"
SAMPLE="${2:-}"
MATCH_SEED="${3:-0}"

# make the config path absolute so it survives the cd into REPO_ROOT
case "$CONFIG" in /*) : ;; *) CONFIG="$SBR/$CONFIG" ;; esac

ARGS=(tstr --config "$CONFIG" --match-seed "$MATCH_SEED")
if [ -n "$SAMPLE" ]; then ARGS+=(--sample "$SAMPLE"); fi

# data_root defaults to CWD; run from the outer repo root so data/ and results_*/ resolve.
( cd "$REPO_ROOT" && python3 "$SBR/preprocess/preprocess.py" "${ARGS[@]}" )

echo
echo "Built 'real', 'synth', and 'synth_matched' under $SBR/data_processed/"
echo "Model integrations are not included; see $SBR/methods/README.md."
echo "ID-method TSTR training dir: $SBR/data_processed/synth_matched"
