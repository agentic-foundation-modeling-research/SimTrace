#!/usr/bin/env bash
# Compare SFT inference backends.
# Run the standalone eval three times — once per backend (vllm, sglang, hf) — and
# report each run's metrics and wall-clock time for comparison.
#
# Edit the variables below (or override via env), then: bash user_model/sft/compare_backends.sh
set -euo pipefail

CONFIG="${CONFIG:-user_model/sft/config.yaml}"
DATA="${DATA:-<huggingface_repo_id>/opera_filtered}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afm-research/workspace/yunan.lu/opera_model/0702/checkpoints}"
LIMIT="${LIMIT:-8}"                 # cap records (small = quick parity check)
TEMPERATURE="${TEMPERATURE:-0.6}"   # 0.0 = greedy (use greedy for parity)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/afm-research/workspace/yunan.lu/opera_model/0702/checkpoints/checkpoint-140/}"

COMMON=(--config "$CONFIG" --data "$DATA" --split "$SPLIT" --output_dir "$OUTPUT_DIR" --adapter_path "$ADAPTER_PATH"
        --limit "$LIMIT" --temperature "$TEMPERATURE" --max_new_tokens "$MAX_NEW_TOKENS")

# --- 1. vLLM ---
echo "=== [1/3] backend=vllm ==="
python -m user_model.inference "${COMMON[@]}" --backend vllm \
    --pred_out "$OUTPUT_DIR/eval_predictions_vllm.jsonl"

# --- 2. SGLang ---
echo "=== [2/3] backend=sglang ==="
python -m user_model.inference "${COMMON[@]}" --backend sglang \
    --pred_out "$OUTPUT_DIR/eval_predictions_sglang.jsonl"

# --- 3. HuggingFace (transformers) ---
echo "=== [3/3] backend=hf ==="
python -m user_model.inference "${COMMON[@]}" --backend hf \
    --pred_out "$OUTPUT_DIR/eval_predictions_hf.jsonl"

# --- Comparison ---
echo
echo "======================= COMPARISON ======================="
printf "%-10s %12s   %s\n" "backend" "gen_time(s)" "metrics"
for b in vllm sglang hf; do
    m="$OUTPUT_DIR/eval_predictions_${b}.metrics.json"
    if [[ -f "$m" ]]; then
        t=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('generation_time_sec','n/a'))" "$m" 2>/dev/null || echo "n/a")
        metrics=$(tr -d '\n' < "$m")
    else
        t="n/a"
        metrics="(missing)"
    fi
    printf "%-10s %12s   %s\n" "$b" "$t" "$metrics"
done
echo "=========================================================="
