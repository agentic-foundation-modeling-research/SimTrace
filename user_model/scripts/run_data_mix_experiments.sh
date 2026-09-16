#!/usr/bin/env bash
# Run SFT real/synthetic data-mix experiments.
#
# Data-mixing experiments: how much does synthetic buyer-sim data help when real OPeRA
# data is scarce (50 sessions)?
#
#   1. opera50_only          — 50 OPeRA sessions
#   2-5. mix_opera50_synN    — 50 OPeRA + sampled synthetic, trained once
#   2-5. seq_synN            — train on sampled synthetic, then continue on 50 OPeRA
#        (two commands: stage 2 passes stage 1's output_dir as --model_name_or_path)
#
# N is the default per-store synthetic session budget (overridable per store below). Every
# run — including sequential stage 1 — is evaluated on
# the *same* OPeRA validation split, so eval_action_f1 is comparable across all 9 and
# stage 1's curve reads directly as synthetic -> real transfer.
#
# OPeRA is still the existing trainer-ready Method-1 dataset. Synthetic data is read from
# the already-sampled normalized repos produced by data_sampling.py and converted
# to trainer-ready SFT JSONL before each selected synthetic experiment.
#
#   python -m user_model.sft.opera_data_prep --config filtered --sample_size 50 \
#       --output_dir data/opera_50 --push_to_hub --repo_id <huggingface_repo_id>/opera_50
#
# The default repo for N is <huggingface_repo_id>/buyer-sim-N. Override SYN_REPO_PREFIX if needed.
#
# Usage:
#   bash user_model/scripts/run_data_mix_experiments.sh                  # all 9 experiments
#   DRY_RUN=1 bash user_model/scripts/run_data_mix_experiments.sh        # print commands only
#   bash user_model/scripts/run_data_mix_experiments.sh mix_opera50_syn500  # a subset
#   FORCE=1 bash user_model/scripts/run_data_mix_experiments.sh          # re-run finished runs
#   EXTRA_ARGS="--max_steps 3 --num_eval_generations 4" bash user_model/... seq_syn50  # smoke
#
# A run whose <output_dir>/trainer_state.json exists is skipped unless FORCE=1, so the
# script is safe to re-invoke after an interruption.

set -uo pipefail

OUT_ROOT="${OUT_ROOT:-/mnt/afm-research/workspace/yunan.lu/user_model/qwen3.5/data_mix}"
NS="${NS:-<huggingface_repo_id>}"                       # HF namespace holding the datasets
OPERA="${OPERA:-$NS/opera_50}"            # the real-data source, also the eval source
SYN_REPO_PREFIX="${SYN_REPO_PREFIX:-$NS/buyer-sim-}" # sampled normalized repos
SYN_SIZES="${SYN_SIZES:-20}"  # synthetic session budgets
DATA_ROOT="${DATA_ROOT:-$OUT_ROOT/prepared_data}"
CONFIG="${CONFIG:-user_model/sft/config.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-user_model/sft/accelerate_config.yaml}"
GPUS="${GPUS:-0,1}"
STAGE2_LR="${STAGE2_LR:-1e-5}"            # lower than the base LR: refine, don't overwrite
EXTRA_ARGS="${EXTRA_ARGS:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-user_model_data_mix}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Run names given as positional args select a subset; no args means run everything.
SELECTED=("$@")

selected() {
  [[ ${#SELECTED[@]} -eq 0 ]] && return 0
  local want
  for want in "${SELECTED[@]}"; do
    # An experiment name selects all its stages ("seq_syn50" -> both stages), and a stage
    # name selects just that stage ("seq_syn50/stage2_opera").
    [[ "$1" == "$want" || "$1" == "$want"/* ]] && return 0
  done
  return 1
}

# run_train <name> [extra trainer flags...]
# <name> becomes both the output dir (under OUT_ROOT) and the wandb run name.
run_train() {
  local name="$1"; shift
  local out="$OUT_ROOT/$name"

  selected "$name" || return 0

  if [[ -f "$out/trainer_state.json" && "${FORCE:-0}" != 1 ]]; then
    echo "[skip] $name (already finished; FORCE=1 to re-run)"
    return 0
  fi

  local cmd=(accelerate launch --config_file "$ACCEL_CONFIG" -m user_model.sft.trainer
             --config "$CONFIG" --output_dir "$out"
             --wandb_run_name "${name//\//_}" "$@" $EXTRA_ARGS)

  echo "[run ] $name"
  echo "       ${cmd[*]}"
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0

  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_RUN_GROUP="$WANDB_RUN_GROUP" \
    "${cmd[@]}" 2>&1 | tee "$out/train.log"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[fail] $name exited with code $rc"
  fi
  return $rc
}

# prepare_data <N>
# Writes trainer-compatible synthetic/ and mixed/ local sources for this budget.
prepare_data() {
  local n="$1"
  local data_dir="$DATA_ROOT/syn$n"
  local sampled_repo="${SYN_REPO_PREFIX}${n}"

  local cmd=(python -m user_model.sft.data_prep
             --synthetic_repo "$sampled_repo"
             --output_dir "$data_dir")

  echo "[prep] syn$n from $sampled_repo"
  echo "       ${cmd[*]}"
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  "${cmd[@]}"
}

# 1. Real data only — the baseline both setups are measured against.
run_train opera50_only \
    --train_datasets "$OPERA" --eval_dataset "$OPERA"

for N in $SYN_SIZES; do
  SYN="$DATA_ROOT/syn$N"

  # Stage 2 consumes only OPeRA and an existing stage-1 checkpoint, so it does not need
  # synthetic preparation when selected by itself.
  if selected "mix_opera50_syn$N" || selected "seq_syn$N/stage1_syn"; then
    prepare_data "$N" || {
      echo "[fail] data preparation for syn$N"
      exit 1
    }
  fi

  # Setup 1 — trainer concatenates and shuffles OPeRA + synthetic rows.
  run_train "mix_opera50_syn$N" \
      --train_datasets "$SYN" "$OPERA" --eval_dataset "$OPERA"

  # Setup 2 — sequential: synthetic first, then continue fine-tuning that checkpoint on
  # OPeRA.
  stage1="$OUT_ROOT/seq_syn$N/stage1_syn"
  run_train "seq_syn$N/stage1_syn" \
      --train_datasets "$SYN" --eval_dataset "$OPERA"

  # Stage 2 needs stage 1's weights, so skip it when that run never saved a checkpoint.
  if [[ -f "$stage1/config.json" || "${DRY_RUN:-0}" == 1 ]]; then
    run_train "seq_syn$N/stage2_opera" \
        --train_datasets "$OPERA" --eval_dataset "$OPERA" \
        --model_name_or_path "$stage1" --learning_rate "$STAGE2_LR"
  elif selected "seq_syn$N/stage2_opera"; then
    echo "[skip] seq_syn$N/stage2_opera (no checkpoint at $stage1)"
  fi
done

echo
echo "Done. Checkpoints under $OUT_ROOT"
echo "Score each run on the held-out OPeRA test split with:"
echo "  python -m user_model.inference --config $CONFIG --output_dir $OUT_ROOT/<run> \\"
echo "      --data $OPERA --split test"
