#!/usr/bin/env bash
# Train-Synthetic-Test-Real evaluation. For each method, run:

#   baseline  train real           -> test real held-out
#   ID TSTR   train synth_matched  -> test SAME real held-out
#   MM TSTR   train synth_<method>_paper -> test SAME real held-out
#
# Each run trains a checkpoint then infers from it, once PER SEED, logging to
#   results/<method>_<train>_to_<test>_seed<k>.log   (e.g.
#      narm_real_to_real_seed0.log, narm_synth_to_real_seed0.log, ...)
# and dumping per-example test outcomes to
#   results/preds/<method>_<train>_to_<test>_seed<k>.npz
# which scripts/significance.py turns into deltas with bootstrap CIs / p-values.
#
# collect_results.py then renders the table: per-arm mean +/- std over seeds, the
# delta vs real->real with its 95% lower confidence bound, and each arm's
# zero-parameter popularity prior (a gap smaller than the prior gap is not a
# finding -- see analysis/tstr_diagnosis/REPORT.md).

#
# Run scripts/prepare_data.sh first to build the real, full-synthetic, and
# ID-volume-matched directories.
# The multimodal methods (dimo, mmsbr) additionally need
# scripts/prepare_data_mm.sh (-> data_processed/*_{dimo,mmsbr}_*);
# they use only the paper feature setting and log with a dashed tag, e.g.
# dimo-paper_real_to_real_seed0.log.
#
# Usage:
#   scripts/tstr.sh                          # all methods, both conditions, seeds 0 1 2
#   METHODS="narm rain" scripts/tstr.sh
#   METHODS="dimo mmsbr" scripts/tstr.sh
#   SEEDS="0 1 2 3 4" scripts/tstr.sh        # more seeds -> tighter mean +/- std
# Env overrides: METHODS, SEEDS, EPOCHS_NARM, EPOCHS_RESTC,
#                EPOCHS_RAIN, EPOCHS_DIMO, EPOCHS_MMSBR, GPU, DIMO_ARGS,
#                DIMO_TAG (result label only; data always uses paper features)

set -euo pipefail

SBR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$SBR/data_processed"
RESULTS="$SBR/results"
CKPTS="$RESULTS/ckpts"
PREDS="$RESULTS/preds"
mkdir -p "$RESULTS" "$CKPTS" "$PREDS"

METHODS="${METHODS:-narm restc rain dimo mmsbr}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS_NARM="${EPOCHS_NARM:-30}"
EPOCHS_RESTC="${EPOCHS_RESTC:-30}"
EPOCHS_RAIN="${EPOCHS_RAIN:-30}"
EPOCHS_DIMO="${EPOCHS_DIMO:-30}"
EPOCHS_MMSBR="${EPOCHS_MMSBR:-30}"
GPU="${GPU:-0}"
DIMO_EXTRA=()
DIMO_TAG="${DIMO_TAG:-paper}"
if [ -n "${DIMO_ARGS:-}" ]; then
  read -r -a DIMO_EXTRA <<< "$DIMO_ARGS"
fi

# condition tag -> (train_dir, test_dir). test is always the SAME real held-out
# set; every processed dir represents the same test examples in a shared
# vocabulary, which is what lets the arms be compared example-by-example (and
# paired in the bootstrap). Multimodal synth dirs also use synthetic catalog
# text, matching what the simulated buyers saw.
#   real_to_real          : train real
#   synth_to_real         : ID methods use synth_matched; multimodal
#                           methods keep synth_<method>_paper
CONDITIONS=("real_to_real:real:real" \
            "synth_to_real:synth:synth")

# optional CONDS filter, e.g. CONDS="real_to_real synth_to_real"
if [ -n "${CONDS:-}" ]; then
  filtered=()
  for cond in "${CONDITIONS[@]}"; do
    for want in $CONDS; do
      [ "${cond%%:*}" = "$want" ] && filtered+=("$cond")
    done
  done
  [ ${#filtered[@]} -eq 0 ] && { echo "CONDS matched no known condition: $CONDS"; exit 1; }
  CONDITIONS=("${filtered[@]}")
fi

for m in $METHODS; do
  for cond in "${CONDITIONS[@]}"; do
    tag="${cond%%:*}"; rest="${cond#*:}"
    traindir="$DATA_ROOT/${rest%%:*}"; testdir="$DATA_ROOT/${rest#*:}"
    if [ "$tag" = "synth_to_real" ]; then
      case "$m" in
        narm|restc|rain)
          traindir="$DATA_ROOT/synth_matched"
          testdir="$DATA_ROOT/synth_matched"
          ;;
      esac
    fi
    if [ ! -d "$traindir" ]; then
      echo "!!!! skipping $m [$tag]: $traindir does not exist."
      echo "     Rebuild it with scripts/prepare_data.sh."
      continue
    fi
    if [ "$tag" = "synth_to_real" ]; then
      case "$m" in
        narm|restc|rain)
          python3 -c 'import json,sys
r=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2]))
assert s.get("match_synth_volume") == "subsample", "synthetic ID data is not volume matched"
assert s["n_train_examples"] == r["n_train_examples"], "real/synthetic training volumes differ"' \
            "$DATA_ROOT/real/stats.json" "$traindir/stats.json" || {
              echo "ERROR: stale or invalid synth_matched; rerun scripts/prepare_data.sh."
              exit 1
            }
          ;;
      esac
    fi
    for seed in $SEEDS; do
      log="$RESULTS/${m}_${tag}_seed${seed}.log"
      ckpt="$CKPTS/${m}_${tag}_seed${seed}.tar"
      preds="$PREDS/${m}_${tag}_seed${seed}.npz"
      echo "############ $m [$tag] seed=$seed train=$traindir -> $log ############"
      case "$m" in
        narm)
          ( cd "$SBR/methods/narm" && \
            CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --mode train \
              --train_dir "$traindir" --save "$ckpt" --epoch "$EPOCHS_NARM" --seed "$seed" && \
            CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --mode infer \
              --load "$ckpt" --test_dir "$testdir" --seed "$seed" \
              --preds_out "$preds" ) 2>&1 | tee "$log"
          ;;
        rain)
          ( cd "$SBR/methods/rain" && \
            CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --mode train \
              --train_dir "$traindir" --save "$ckpt" --epoch "$EPOCHS_RAIN" --numcuda 0 --seed "$seed" && \
            CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --mode infer \
              --load "$ckpt" --test_dir "$testdir" --numcuda 0 --seed "$seed" \
              --preds_out "$preds" ) 2>&1 | tee "$log"
          ;;
        restc)
          # RESTC uses torch.distributed; launch a single-process group with torchrun.
          ( cd "$SBR/methods/restc" && mkdir -p output_dir && \
            CUDA_VISIBLE_DEVICES="$GPU" torchrun --nproc_per_node=1 --master_port=29556 \
              train_main_global.py --mode train --train_dir "$traindir" \
              --save "$ckpt" --epoch "$EPOCHS_RESTC" --output_dir output_dir/ --seed "$seed" && \
            CUDA_VISIBLE_DEVICES="$GPU" torchrun --nproc_per_node=1 --master_port=29556 \
              train_main_global.py --mode infer --load "$ckpt" \
              --test_dir "$testdir" --output_dir output_dir/ --seed "$seed" \
              --preds_out "$preds" ) 2>&1 | tee "$log"
          ;;
        dimo|mmsbr)
          # multimodal paper setting; data dirs carry a _<method>_paper suffix
          # and the log tag uses a dash so
          # collect_results.py's split("_", 1) yields method "dimo-paper".
          epochs="$EPOCHS_DIMO"; [ "$m" = "mmsbr" ] && epochs="$EPOCHS_MMSBR"
          extra_args=()
          [ "$m" = "dimo" ] && extra_args=("${DIMO_EXTRA[@]}")
          for v in paper; do
            result_variant="$v"
            [ "$m" = "dimo" ] && result_variant="$DIMO_TAG"
            vlog="$RESULTS/${m}-${result_variant}_${tag}_seed${seed}.log"
            vckpt="$CKPTS/${m}-${result_variant}_${tag}_seed${seed}.tar"
            vpreds="$PREDS/${m}-${result_variant}_${tag}_seed${seed}.npz"
            vtrain="${traindir}_${m}_${v}"
            vtest="${testdir}_${m}_${v}"
            echo "############ $m-$v [$tag] seed=$seed train=$vtrain -> $vlog ############"
            ( cd "$SBR/methods/$m" && \
              CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --mode train \
                --train_dir "$vtrain" --save "$vckpt" --epoch "$epochs" --seed "$seed" \
                "${extra_args[@]}" && \
              CUDA_VISIBLE_DEVICES="$GPU" python3 main.py --mode infer \
                --load "$vckpt" --test_dir "$vtest" --seed "$seed" \
                --preds_out "$vpreds" "${extra_args[@]}" ) 2>&1 | tee "$vlog"
          done
          ;;
        *) echo "unknown method $m"; exit 1 ;;
      esac
    done
  done
done

echo "############ collecting results ############"
python3 "$SBR/scripts/collect_results.py" --results "$RESULTS"

echo "############ significance (paired cluster bootstrap) ############"
python3 "$SBR/scripts/significance.py" --preds "$PREDS" \
  --json "$RESULTS/significance.json"
