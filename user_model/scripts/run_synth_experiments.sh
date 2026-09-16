#!/usr/bin/env bash
# Run synthetic-only SFT experiments.
#
# Train synthetic-only SFT models on the full contents of each prepared
# buyer-sim dataset.
#
# Usage:
#   bash user_model/scripts/run_synth_experiments.sh
#   DRY_RUN=1 bash user_model/scripts/run_synth_experiments.sh
#   bash user_model/scripts/run_synth_experiments.sh full_50 full_500
#   FORCE=1 bash user_model/scripts/run_synth_experiments.sh
#
# A completed output (one containing trainer_state.json) is skipped unless FORCE=1.

set -uo pipefail

SFT_ROOT="${SFT_ROOT:-/mnt/afm-research/workspace/yunan.lu/user_model/qwen3.5/sft_validate_real}"
DATA_ROOT="${DATA_ROOT:-$SFT_ROOT/data}"
FULL_ROOT="${FULL_ROOT:-$SFT_ROOT/full}"
SYN_SIZES="${SYN_SIZES:-50 100 250 500}"

CONFIG="${CONFIG:-user_model/sft/config.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-user_model/sft/accelerate_config.yaml}"
GPUS="${GPUS:-0,1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-sft_synth}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SELECTED=("$@")
EXTRA_ARGS_ARRAY=()
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"
fi

selected() {
  [[ ${#SELECTED[@]} -eq 0 ]] && return 0
  local requested
  for requested in "${SELECTED[@]}"; do
    [[ "$1" == "$requested" ]] && return 0
  done
  return 1
}

# run_train <dataset-size>
run_train() {
  local size="$1"
  local name="full_${size}"
  local dataset="$DATA_ROOT/buyer-sim-$size"

  selected "$name" || return 0

  local output_dir="$FULL_ROOT/checkpoints_$size"
  if [[ -f "$output_dir/trainer_state.json" && "${FORCE:-0}" != 1 ]]; then
    echo "[skip] $name (already finished; FORCE=1 to re-run)"
    return 0
  fi

  if [[ "${DRY_RUN:-0}" != 1 && ! -s "$dataset/sft_train.jsonl" ]]; then
    echo "[fail] missing or empty training file: $dataset/sft_train.jsonl" >&2
    return 1
  fi

  local cmd=(
    accelerate launch
    --config_file "$ACCEL_CONFIG"
    -m user_model.sft.trainer
    --config "$CONFIG"
    --train_datasets "$dataset"
    --output_dir "$output_dir"
    --wandb_run_name "synth_sft_${name}"
  )
  if [[ -n "$EXTRA_ARGS" ]]; then
    cmd+=("${EXTRA_ARGS_ARRAY[@]}")
  fi

  echo "[run ] $name"
  echo "       dataset: $dataset"
  echo "       output:  $output_dir"
  echo "       ${cmd[*]}"
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0

  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_RUN_GROUP="$WANDB_RUN_GROUP" \
    "${cmd[@]}" 2>&1 | tee "$output_dir/train.log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[fail] $name exited with code $rc" >&2
  fi
  return "$rc"
}

for size in $SYN_SIZES; do
  run_train "$size" || exit $?
done

echo "Done. Full-data checkpoints: $FULL_ROOT/checkpoints_{N}"
