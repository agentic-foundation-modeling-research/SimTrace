#!/usr/bin/env bash
# Evaluate SFT real/synthetic data-mix experiments.
#
# Score every data-mixing checkpoint on the held-out OPeRA test split, then print a ranked
# summary. Companion to run_data_mix_experiments.sh — same OUT_ROOT, same run names.
#
# Each run is the equivalent of:
#
#   python -m user_model.inference --config user_model/sft/config.yaml \
#       --data <huggingface_repo_id>/opera_filtered --split test \
#       --adapter_path <run_dir> --output_dir <run_dir>
#
# i.e. load the model saved at <run_dir> and write eval_predictions.jsonl +
# eval_predictions.metrics.json back into that same folder.
#
# 13 checkpoints are scored: opera50_only, the 4 mix_* runs, and both stages of each seq_*
# run. The seq stage-1 rows are the synthetic-only points (no real data at all), so the
# summary shows the whole picture: real-only, mixed, synthetic-only, and synthetic->real.
#
# Usage:
#   bash user_model/scripts/run_data_mix_inference.sh                    # score everything
#   DRY_RUN=1 bash user_model/scripts/run_data_mix_inference.sh          # print commands only
#   bash user_model/scripts/run_data_mix_inference.sh seq_syn100         # a subset, by name
#   bash user_model/scripts/run_data_mix_inference.sh seq_syn100/stage2_opera  # one stage
#   FORCE=1 bash user_model/scripts/run_data_mix_inference.sh            # re-score finished runs
#   SUMMARY_ONLY=1 bash user_model/scripts/run_data_mix_inference.sh     # rebuild the table
#   AUTO_CKPT=1 bash user_model/scripts/run_data_mix_inference.sh        # let inference pick the
#                                                                # best checkpoint-N instead
#
# A run whose eval_predictions.jsonl already exists is skipped unless FORCE=1.
#
# The summary is written to eval_summary.csv.

set -uo pipefail

OUT_ROOT="${OUT_ROOT:-/mnt/afm-research/workspace/yunan.lu/user_model/qwen3.5/data_mix}"
TEST_DATA="${TEST_DATA:-<huggingface_repo_id>/opera_filtered}"   # held-out real sessions
SPLIT="${SPLIT:-test}"
SYN_SIZES="${SYN_SIZES:-10 20 50 100}"
CONFIG="${CONFIG:-user_model/sft/config.yaml}"
GPUS="${GPUS:-0,1}"
BACKEND="${BACKEND:-vllm}"
# Shard across every GPU listed in GPUS unless told otherwise.
TP_SIZE="${TP_SIZE:-$(awk -F, '{print NF}' <<< "$GPUS")}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SELECTED=("$@")

selected() {
  [[ ${#SELECTED[@]} -eq 0 ]] && return 0
  local want
  for want in "${SELECTED[@]}"; do
    # An experiment name selects all its stages; a stage name selects just that stage.
    [[ "$1" == "$want" || "$1" == "$want"/* ]] && return 0
  done
  return 1
}

# All the run names this script knows about, in report order.
RUN_NAMES=(opera50_only)
for N in $SYN_SIZES; do
  RUN_NAMES+=("mix_opera50_syn$N" "seq_syn$N/stage1_syn" "seq_syn$N/stage2_opera")
done

# run_eval <name> — score the checkpoint at $OUT_ROOT/<name> against the test split.
run_eval() {
  local name="$1"
  local out="$OUT_ROOT/$name"

  selected "$name" || return 0

  if [[ ! -f "$out/config.json" && "${DRY_RUN:-0}" != 1 ]]; then
    echo "[skip] $name (no checkpoint at $out)"
    return 0
  fi
  if [[ -f "$out/eval_predictions.jsonl" && "${FORCE:-0}" != 1 ]]; then
    echo "[skip] $name (already scored; FORCE=1 to re-score)"
    return 0
  fi

  local cmd=(python -m user_model.inference --config "$CONFIG"
             --data "$TEST_DATA" --split "$SPLIT"
             --output_dir "$out"
             --backend "$BACKEND" --tp_size "$TP_SIZE"
             --max_new_tokens "$MAX_NEW_TOKENS")
  # Default: evaluate exactly the model saved at output_dir (which is the best checkpoint,
  # since training runs with load_best_model_at_end). AUTO_CKPT=1 drops --adapter_path and
  # lets inference resolve best_model_checkpoint from trainer_state.json instead.
  [[ "${AUTO_CKPT:-0}" != 1 ]] && cmd+=(--adapter_path "$out")
  cmd+=($EXTRA_ARGS)

  echo "[eval] $name"
  echo "       ${cmd[*]}"
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0

  CUDA_VISIBLE_DEVICES="$GPUS" "${cmd[@]}" 2>&1 | tee "$out/eval.log"

  local rc=${PIPESTATUS[0]}
  [[ $rc -ne 0 ]] && echo "[fail] $name exited with code $rc"
  return $rc
}

if [[ "${SUMMARY_ONLY:-0}" != 1 ]]; then
  for name in "${RUN_NAMES[@]}"; do
    run_eval "$name"
  done
fi

[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

# --- Summary -------------------------------------------------------------------------- #
# Collect every run's eval_predictions.metrics.json into one table + CSV.
OUT_ROOT="$OUT_ROOT" RUN_NAMES="${RUN_NAMES[*]}" TEST_DATA="$TEST_DATA" python - <<'PY'
import csv, json, os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
names = os.environ["RUN_NAMES"].split()
keys = ["exact_match_acc", "action_acc", "action_f1", "action_f1_weighted", "num_turns"]

rows = []
for name in names:
    path = root / name / "eval_predictions.metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"run": name, **{k: m.get(k) for k in keys}})

if not rows:
    print(f"\nNo metrics found under {root} yet.")
    raise SystemExit(0)

print(f"\n=== OPeRA test-split results ({os.environ['TEST_DATA']}), ranked by exact_match_acc ===")
header = f"{'run':<26} {'exact_match':>12} {'action_acc':>11} {'action_f1':>10} {'turns':>7}"
print(header)
print("-" * len(header))
for r in sorted(rows, key=lambda r: r.get("exact_match_acc") or 0.0, reverse=True):
    def fmt(key, width, prec=4):
        v = r.get(key)
        return f"{v:>{width}.{prec}f}" if isinstance(v, float) else f"{'-':>{width}}"
    print(f"{r['run']:<26} {fmt('exact_match_acc', 12)} {fmt('action_acc', 11)} "
          f"{fmt('action_f1', 10)} {str(r.get('num_turns', '-')):>7}")

out_csv = root / "eval_summary.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["run"] + keys)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWrote {out_csv}")
PY

echo
echo "Per-turn predictions: $OUT_ROOT/<run>/eval_predictions.jsonl"
echo "Point estimates:      $OUT_ROOT/eval_summary.csv"
echo "Diff two runs with:"
echo "  python -m user_model.compare_predictions \\"
echo "      --file1 $OUT_ROOT/opera50_only/eval_predictions.jsonl \\"
echo "      --file2 $OUT_ROOT/mix_opera50_syn10/eval_predictions.jsonl \\"
echo "      --label1 opera50_only --label2 mix_syn10 --out-dir analysis/data_mix"
