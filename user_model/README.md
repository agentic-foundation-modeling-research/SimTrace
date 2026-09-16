# User model training

This package reproduces the next-action prediction pipeline described in
Section 4.2.2 (Low-Resource Setting) of the SIMTRACE paper.
Run all commands from the repository root. The dataset currently is anonymized, will be released soon.

## 1. Install dependencies
The following training was conducted on a x86_64 server with Ubuntu 24.04.2 LTS system and 8 GPUs:
NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0 

```bash
conda create -n buyer-sim-vllm python=3.12
conda activate buyer-sim-vllm
pip install -r user_model/requirements.txt
```

## 2. Prepare SFT data

Sample ten sessions from each of the five stores in the complete synthetic
source. `<huggingface_repo_id>/buyer-sim-complete-v4` and its verified store UUIDs are the
defaults, so neither needs to be repeated on the command line:

```bash
python -m user_model.sft.data_sampling \
  --synthetic_repo <huggingface_repo_id>/buyer-sim-complete-v4 \
  --output_repo <huggingface_repo_id>/buyer-sim-250-test \
  --sample_size 50 \
  --private
```

To publish cumulative nested tiers, provide matching repository and size lists:

```bash
python -m user_model.sft.data_sampling \
  --synthetic_repo <huggingface_repo_id>/buyer-sim-complete-v4 \
  --output_repo <huggingface_repo_id>/buyer-sim-50 <huggingface_repo_id>/buyer-sim-100 \
  --sample_size 10 20 \
  --private
```

Each size is per store. With a single output repository, `--sample_size` may
also contain one value per explicitly selected `--store_id`.

Convert a normalized synthetic dataset (the `action` and `user` Hub configs)
to multi-turn training records:

```bash
python -m user_model.sft.data_prep \
  --synthetic_repo <huggingface_repo_id>/buyer-sim-250-test \
  --output_dir data/sft/buyer-sim-250
```

Prepare the 50-session real OPeRA arm while keeping validation and test fixed:

```bash
python -m user_model.sft.opera_data_prep \
  --config filtered \
  --sample_size 50 \
  --output_dir data/sft/opera-50 \
  --push_to_hub \
  --repo_id <huggingface_repo_id>/opera-50-sft
```

Each local prepared directory contains `sft_train.jsonl` and its available
validation/test files. Data splits are made at session level before sliding
windows are expanded.

## 3. Run SFT

Set `train_datasets` and `eval_dataset` in `user_model/sft/config.yaml`, or
override them at launch. Examples for the three paper arms are:

```bash
# Synthetic only (this checkpoint initializes RL).
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file user_model/sft/accelerate_config.yaml -m user_model.sft.trainer \
  --config user_model/sft/config.yaml \
  --train_datasets data/sft/buyer-sim-250 \
  --eval_dataset <huggingface_repo_id>/opera-50-sft \
  --output_dir ckpt/sft/synth

# Real only.
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file user_model/sft/accelerate_config.yaml -m user_model.sft.trainer \
  --config user_model/sft/config.yaml \
  --train_datasets <huggingface_repo_id>/opera-50 \
  --eval_dataset <huggingface_repo_id>/opera-50-sft \
  --output_dir ckpt/sft/real

# Real + synthetic augmentation.
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file user_model/sft/accelerate_config.yaml -m user_model.sft.trainer \
  --config user_model/sft/config.yaml \
  --train_datasets data/sft/buyer-sim-250 <huggingface_repo_id>/opera-50 \
  --eval_dataset <huggingface_repo_id>/opera-50-sft \
  --output_dir ckpt/sft/real-plus-synth
```


The experiment helpers `user_model/scripts/run_data_mix_experiments.sh` and
`user_model/scripts/run_data_mix_inference.sh` run and evaluate the larger
real/synthetic matrix.

## 4. Prepare the real-session RL data

Recover the exact OPeRA sessions represented by the 50-session SFT dataset and
publish them in the normalized `action`/`user` layout:

```bash
python -m user_model.rl.opera_data_prep \
  --reference-repo <huggingface_repo_id>/opera-50-sft \
  --reference-split train \
  --output-repo <huggingface_repo_id>/opera-50-rl \
  --push-to-hub
```

Flatten those real conversations to next-action records:

```bash
python -m user_model.rl.data_prep \
  --input-repo <huggingface_repo_id>/opera-50-rl \
  --input-split train \
  --output-dir data/rl/opera-50
```

This produces `data/rl/opera-50/train.jsonl`. Point `train_data_path` in
`user_model/rl/config.yaml` to that file and set `model_name_or_path` to the
synthetic-only SFT checkpoint from Step 3.

## 5. Run RL

With server-mode vLLM, reserve separate devices for rollout generation and
training. Start the rollout server first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m user_model.rl.vllm_server \
  --model ckpt/sft/synth \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32128 \
  --dtype bfloat16 \
  --enable-prefix-caching true \
  --port 8000
```

Then launch four-process ZeRO-3 training on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
accelerate launch --config_file user_model/rl/accelerate_config.yaml \
  -m user_model.rl.trainer \
  --config user_model/rl/config.yaml
```

## 6. Evaluate

Use the same held-out OPeRA test split for every SFT/RL checkpoint so the three
paper metrics remain comparable:

```bash
python -m user_model.inference \
  --config user_model/sft/config.yaml \
  --data data/sft/opera-50/sft_eval.jsonl \
  --split test \
  --adapter_path <checkpoint-or-adapter-path>
```
Notice: both sft and sft+rl use --data input from sft. Either "--data <huggingface_repo_id>/opera-50-sft --split test" or "--data data/sft/opera-50/sft_eval.jsonl".


## Layout

```text
user_model/
├── README.md
├── inference.py           # shared checkpoint inference and evaluation
├── prompt_templates.py    # prompt and action formatting shared by SFT and RL
├── sft/
│   ├── data_prep.py       # normalized sessions -> multi-turn SFT JSONL
│   ├── data_loader.py     # shared Session/Step records and window conversion
│   ├── trainer.py         # TRL SFT entry point
│   ├── config.yaml        # Qwen3.5-9B SFT configuration
│   └── ...                # sampling, inference, evaluation, and launch helpers
└── rl/
    ├── data_prep.py       # conversations -> next-action turn records
    ├── trainer.py         # offline GDPO/GRPO entry point
    ├── config.yaml        # reward, rollout, and optimization configuration
    └── ...                # rewards, OPeRA recovery, vLLM, and test helpers
```