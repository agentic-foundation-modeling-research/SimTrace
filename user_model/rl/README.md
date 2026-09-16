# RL details

See `user_model/README.md` for the complete paper-aligned SFT-to-RL pipeline.
This page documents RL reward and launch details.

The trainer supports two advantage formulas over the same three reward
components:

```text
R_format = valid output JSON/action schema
R_action = exact action-type match
R_target = 0.60 * LCS(pred_target, gold_target)
           + 0.10 * valid_UI_target
           + 0.30 * exact_target
```
GDPO still receives `R_target` as one reward component. `LCS` is the longest
common contiguous substring length divided by the average target length.
`valid_UI_target` is 1 when the predicted target is a semantic ID in the
current UI, and `exact_target` is 1 when predicted and gold targets match.

Select the formula in YAML with `advantage_formula`, or override it on the
command line:

```bash
python -m user_model.rl.trainer \
  --config user_model/rl/config.yaml \
  --advantage-formula gdpo
```

`gdpo` passes the three components to TRL as separate reward functions. TRL
1.7.0 natively implements GDPO through
`multi_objective_aggregation="normalize_then_sum"`: it normalizes each reward
within each prompt's generation group, sums those normalized objectives, and
normalizes the combined result across the generation batch before using it as
the final advantage.

### Other reward option
`constraint-aware` computes one scalar reward before TRL's standard group
normalization:

```text
R = -w_format * (1 - R_format)
    + R_format * (w_action * R_action + w_target * R_target)
```

The weights live under `reward:` in `config.yaml`, and validation
requires `w_action + w_target = 1`.

```bash
# Sample complete sessions once and publish the normalized subset for reuse by
# both SFT and RL formatting pipelines.
python -m user_model.sft.data_sampling \
  --output_repo <huggingface_repo_id>/buyer-sim-50 \
  --sample_size 10 \
  --private

# Convert every valid session in the same sampled repo to Method-1 SFT records.
python -m user_model.sft.data_prep \
  --synthetic_repo <huggingface_repo_id>/buyer-sim-50 \
  --output_dir data/sft/buyer-sim-50

# This writes data/sft/buyer-sim-50/sft_train.jsonl and manifest.json.

# Convert every valid session in the sampled repo to local RL turn records.
python -m user_model.rl.data_prep \
  --input-repo <huggingface_repo_id>/buyer-sim-50 \
  --input-split train+test \
  --output_dir /mnt/afm-research/workspace/yunan.lu/user_model/qwen3.5/rl/data/buyer-sim-50

# This writes data/rl/buyer-sim-50/train.jsonl and manifest.json. Point train_data_path
# at that local JSONL in config.yaml, then start the rollout server.
# Qwen3.5-9B has 16 attention heads, so TP=4 is supported while TP=6 is not.
# Use the local wrapper so multiprocessing selects spawn before CUDA is probed.
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m user_model.rl.vllm_server \
  --model /mnt/afm-research/workspace/yunan.lu/user_model/qwen3.5/sft_validate_real/full/checkpoints_250/checkpoint-20 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32128 \
  --dtype bfloat16 \
  --enable-prefix-caching true \
  --port 8000

# In a second shell, reserve GPUs 4-7 for full-parameter ZeRO-3 training.
# TRL pushes the current policy weights to the rollout server before the first
# generation following every optimizer update.
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_RUN_GROUP="rl" \
accelerate launch --config_file user_model/rl/accelerate_config.yaml \
  -m user_model.rl.trainer \
  --config user_model/rl/config.yaml
```
