"""SFT trainer entry point for the buyer simulation model.

Uses the current TRL API:
  - SFTConfig(assistant_only_loss=True, max_length=...) handles assistant-only loss
    and chat-template application internally for Qwen3.
  - SFTTrainer takes a HF Dataset of {"messages": [...]} records directly.

Quick start
-----------
# 1) Convert a sampled normalized Hub dataset to the configured local SFT source.
python -m user_model.sft.data_prep \
    --synthetic_repo <huggingface_repo_id>/buyer-sim-50 \
    --output_dir data/sft/buyer-sim-50

# 2) Train with 2 GPUs (recommended — edit num_processes in accelerate_config.yaml to match your setup)
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --config_file user_model/sft/accelerate_config.yaml \
  -m user_model.sft.trainer --config user_model/sft/config.yaml \
  --train_datasets data/sft/buyer-sim-50
  --output_dir ckpt/sft/buyer-sim-50

# To train on mixed sampled synthetic data and OPeRA, pass both sources. 
python -m user_model.sft.trainer --config user_model/sft/config.yaml \
    --train_datasets data/sft/buyer-sim-50 <huggingface_repo_id>/opera_50 \
    --eval_dataset <huggingface_repo_id>/opera_50

# Sequential training: synthetic first, then continue on real data from that checkpoint
python -m user_model.sft.trainer --config user_model/sft/config.yaml \
    --train_datasets data/sft/buyer-sim-50 --eval_dataset <huggingface_repo_id>/opera_50 \
    --output_dir ckpt/sft/stage1_synth
python -m user_model.sft.trainer --config user_model/sft/config.yaml \
    --train_datasets <huggingface_repo_id>/opera_50 --eval_dataset <huggingface_repo_id>/opera_50 \
    --model_name_or_path ckpt/sft/stage1_synth \
    --output_dir ckpt/sft/stage2_real --learning_rate 1e-5

"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from sklearn.metrics import f1_score as sklearn_f1

import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)

# transformers 5.x names the vision-language auto-class AutoModelForMultimodalLM
# (4.x called it AutoModelForImageTextToText); guarded so this module still imports
# on a version that lacks it.
try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_LOG_SAMPLES = 8  # number of prediction samples to print each eval


def _parse_method_target(text: str) -> tuple[str, str]:
    try:
        obj = json.loads(text)
        action = obj.get("action", "")
        if isinstance(action, dict):
            action = action.get("method", "")
            target = action.get("target", "")
        else:
            target = obj.get("target", "")
        return action, target
    except (json.JSONDecodeError, TypeError, ValueError):
        return "", ""


def _input_ids(encoded) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids", [])
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def _rendered_length(
    tokenizer,
    messages: list[dict],
    cap: int | None = None,
) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenize_kwargs = {}
    if cap is not None:
        # Probe with a cap instead of tokenizing the entire DOM.
        tokenize_kwargs = {"truncation": True, "max_length": cap}
    return len(_input_ids(tokenizer(rendered, **tokenize_kwargs)))


def _encode_content(tokenizer, content: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(content, add_special_tokens=False))
    return _input_ids(tokenizer(content, add_special_tokens=False))


def _truncate_latest_user(
    tokenizer,
    system: list[dict],
    latest_user: dict,
    budget: int,
) -> list[dict]:
    """Keep the prefix of an oversized current observation within ``budget``."""
    content = latest_user.get("content", "")
    context_header = "# context\n" if content.startswith("# context\n") else ""
    body = content[len(context_header):]
    body_ids = _encode_content(tokenizer, body)

    def candidate(token_count: int) -> list[dict]:
        retained_prefix = tokenizer.decode(
            body_ids[:token_count],
            skip_special_tokens=True,
        )
        user = dict(latest_user)
        user["content"] = context_header + retained_prefix
        return system + [user]

    minimum = 0 if context_header else 1
    if len(body_ids) < minimum:
        raise ValueError("current user query has no tokenizable content")
    minimum_length = _rendered_length(tokenizer, candidate(minimum))
    if minimum_length > budget:
        raise ValueError(
            "system message plus the minimum current user query exceeds "
            f"the {budget}-token prompt budget"
        )

    token_count = min(len(body_ids), minimum + budget - minimum_length)
    while True:
        result = candidate(token_count)
        rendered_length = _rendered_length(tokenizer, result)
        if rendered_length <= budget:
            return result
        token_count -= max(1, rendered_length - budget)
        if token_count < minimum:
            raise ValueError(
                "could not fit the minimum current user query within "
                f"the {budget}-token prompt budget"
            )


def _left_truncate_messages(tokenizer, messages: list[dict], budget: int) -> list[dict]:
    """Fit a prompt while always retaining its latest user observation."""
    if budget <= 0:
        raise ValueError("prompt token budget must be positive")
    if not messages:
        raise ValueError("prompt messages must not be empty")
    if _rendered_length(tokenizer, messages, cap=budget + 1) <= budget:
        return messages

    system = [message for message in messages if message.get("role") == "system"]
    history = [message for message in messages if message.get("role") != "system"]
    if not history or history[-1].get("role") != "user":
        raise ValueError("generation prompt must end with the current user query")

    # Remove older complete exchanges without leaving an orphaned assistant turn.
    for start, message in enumerate(history):
        if start == 0 or message.get("role") != "user":
            continue
        candidate = system + history[start:]
        if _rendered_length(tokenizer, candidate, cap=budget + 1) <= budget:
            return candidate

    minimal = system + [history[-1]]
    if _rendered_length(tokenizer, minimal, cap=budget + 1) <= budget:
        return minimal
    return _truncate_latest_user(tokenizer, system, history[-1], budget)


def _iter_assistant_turns(
    messages: list[dict],
    tokenizer=None,
    max_length: Optional[int] = None,
    max_new_tokens: int = 0,
):
    """Yield (prompt_messages, true_text) for every assistant turn in the conversation.

    prompt_messages is all prior ground-truth messages (left-truncated to fit within
    max_length - max_new_tokens tokens when tokenizer/max_length are given); true_text
    is the assistant response.
    """
    budget = None if max_length is None else max_length - max_new_tokens
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            prompt_msgs = messages[:i]
            if tokenizer is not None and budget is not None:
                prompt_msgs = _left_truncate_messages(tokenizer, prompt_msgs, budget)
            yield prompt_msgs, msg["content"]


class SFTTrainerWithGenEval(SFTTrainer):
    def __init__(self, *args, gen_eval_data, num_eval_generations, eval_max_new_tokens,
                 eval_gen_batch_size=16, eval_temperature=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_eval_data = gen_eval_data
        self._num_eval_gen = num_eval_generations
        self._eval_max_new_tokens = eval_max_new_tokens
        self._eval_gen_batch_size = eval_gen_batch_size
        self._eval_temperature = eval_temperature

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        torch.cuda.empty_cache()
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self._gen_eval_data:
            gen_metrics = self._run_gen_eval(metric_key_prefix)
            metrics.update(gen_metrics)
            self.log(gen_metrics)
            # super().evaluate() already fired on_evaluate with the base metrics dict
            # (before gen_metrics existed), so EarlyStoppingCallback never saw
            # eval_action_f1. Re-fire it now with the complete dict.
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _run_gen_eval(self, prefix: str) -> dict:
        # empty CUDA cache to maximize available memory for generation
        torch.cuda.empty_cache()

        # model = self.model
        model = self.accelerator.unwrap_model(self.model)
        tokenizer = self.processing_class
        samples = (
            self._gen_eval_data
            if self._num_eval_gen in (None, -1)
            else self._gen_eval_data[:self._num_eval_gen]
        )

        # Flatten sampled records into a flat list of (prompt_str, true_text) turns.
        prompts, true_texts = [], []
        for record in samples:
            for prompt_msgs, true_text in _iter_assistant_turns(
                record["messages"], tokenizer=tokenizer,
                max_length=self.args.max_length, max_new_tokens=self._eval_max_new_tokens,
            ):
                if not true_text:
                    continue
                prompts.append(tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                ))
                true_texts.append(true_text)

        # Sort by length so each batch pads to a similar size (minimizes wasted compute),
        # keeping the original index so predictions realign with their true_text.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

        prev_use_cache = model.config.use_cache
        prev_padding_side = tokenizer.padding_side
        model.config.use_cache = True
        tokenizer.padding_side = "left"  # decoder-only batched generation requires left padding
        model.eval()

        # temperature <= 0 → greedy; > 0 → sampling at that temperature.
        gen_kwargs = (
            {"do_sample": True, "temperature": self._eval_temperature}
            if self._eval_temperature and self._eval_temperature > 0
            else {"do_sample": False}
        )

        pred_texts: list[Optional[str]] = [None] * len(prompts)
        bs = max(1, self._eval_gen_batch_size)
        try:
            for start in range(0, len(order), bs):
                batch_idx = order[start:start + bs]
                batch_prompts = [prompts[i] for i in batch_idx]
                model_device = getattr(model, "device", None)
                if model_device is None:
                    model_device = next(model.parameters()).device
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to(model_device)
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=self._eval_max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        **gen_kwargs,
                    )
                # Left padding makes every row's prompt end at the same column,
                # so new tokens start at input_len for the whole batch.
                input_len = inputs["input_ids"].shape[1]
                new_ids = out_ids[:, input_len:]
                decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                for i, text in zip(batch_idx, decoded):
                    pred_texts[i] = text
        finally:
            model.config.use_cache = prev_use_cache
            tokenizer.padding_side = prev_padding_side

        pred_actions, true_actions, exact_hits, log_rows = [], [], [], []
        for pred_text, true_text in zip(pred_texts, true_texts):
            pm, pt = _parse_method_target(pred_text)
            tm, tt = _parse_method_target(true_text)
            if not tm:
                continue
            pred_actions.append(pm)
            true_actions.append(tm)
            exact_hits.append(pm == tm and pt == tt)
            log_rows.append((pred_text, true_text, pm, tm))

        logger.info("=== Gen-eval samples (%d total) ===", len(log_rows))
        for idx, (pred_text, true_text, pm, tm) in enumerate(log_rows[:_LOG_SAMPLES]):
            marker = "✓" if pm == tm else "✗"
            logger.info(
                "[%d] %s  pred=%r  true=%r\n      PRED: %s\n      TRUE: %s",
                idx, marker, pm, tm, pred_text[:300], true_text[:300],
            )

        if not true_actions:
            return {f"{prefix}_exact_match_acc": 0.0, f"{prefix}_action_acc": 0.0, f"{prefix}_action_f1": 0.0}

        n = len(true_actions)
        result = {
            f"{prefix}_exact_match_acc": sum(exact_hits) / n,
            f"{prefix}_action_acc": sum(p == t for p, t in zip(pred_actions, true_actions)) / n,
            f"{prefix}_action_f1": sklearn_f1(true_actions, pred_actions, average="macro", zero_division=0),
        }
        logger.info(
            "=== Gen-eval metrics: exact_match_acc=%.4f  action_acc=%.4f  action_f1=%.4f ===",
            result[f"{prefix}_exact_match_acc"],
            result[f"{prefix}_action_acc"],
            result[f"{prefix}_action_f1"],
        )
        return result


@dataclass
class TrainConfig:
    # Legacy single-source fields. train_datasets / eval_dataset are preferred.
    train_data_path: str = "data/sft/sft_train.jsonl"
    val_data_path: Optional[str] = "data/sft/sft_val.jsonl"
    # When set, load train/validation splits from this HF Hub dataset id instead of the local paths.
    dataset_hub_id: Optional[str] = None

    # Train on the concatenation of these sources (Hub ids, local dirs, or .jsonl files).
    # Takes precedence over dataset_hub_id / train_data_path when non-empty.
    train_datasets: Optional[list[str]] = None
    # Source whose validation split drives eval / early stopping / best-checkpoint selection.
    # Defaults to the first entry of train_datasets.
    eval_dataset: Optional[str] = None

    model_name_or_path: str = "Qwen/Qwen3-14B"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_double_quant: bool = True

    # Fine-tuning mode: "lora" (QLoRA/LoRA adapters) or "full" (all parameters).
    # "full" is incompatible with 4-bit quantization — use_4bit is forced off.
    finetuning_type: str = "lora"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    max_length: int = 32768

    # Kernel/attention backends — GPU defaults; override to false/"eager" for CPU/MPS smoke runs.
    attn_implementation: str = "flash_attention_2"
    use_liger_kernel: bool = True
    assistant_only_loss: bool = True

    output_dir: str = "user_model/sft/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: Optional[bool] = None
    optim: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 4

    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: Optional[int] = 3
    eval_strategy: str = "steps"
    eval_steps: int = 100
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_action_f1"
    greater_is_better: Optional[bool] = None

    # Early stopping (monitors metric_for_best_model; requires eval + load_best_model_at_end).
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    report_to: str = "none"
    wandb_project: str = "buyer-sim-gen"
    wandb_run_name: Optional[str] = None

    # Number of validation records to run generation-based eval on.
    # null / -1 means the entire validation set (lower variance, but slower eval).
    num_eval_generations: Optional[int] = None
    eval_max_new_tokens: int = 256
    # Number of assistant turns generated together per model.generate call during gen-eval.
    eval_gen_batch_size: int = 16
    # Decoding temperature for gen-eval. 0.0 = greedy (deterministic); > 0 enables sampling.
    eval_temperature: float = 0.0

    seed: int = 42
    max_steps: int = -1


def load_config(path: str) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = TrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            logger.warning("Unknown config key: %s", k)
    return cfg


def _bnb_config(cfg: TrainConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_double_quant,
    )


_MULTIMODAL_ARCH_KEYWORDS = ("VL", "Vision", "ImageText", "Omni", "Multimodal")


def _is_multimodal(config) -> bool:
    """True if the model needs the multimodal auto-class + processor rather than a plain causal LM."""
    if getattr(config, "vision_config", None) is not None:
        return True
    archs = getattr(config, "architectures", None) or []
    return any(any(kw in a for kw in _MULTIMODAL_ARCH_KEYWORDS) for a in archs)


def build_model(model_path: str, **kwargs):
    """Load a model, auto-selecting the causal-LM vs multimodal class from its config."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if _is_multimodal(config):
        if AutoModelForMultimodalLM is None:
            raise RuntimeError(
                f"{model_path} is multimodal but AutoModelForMultimodalLM is unavailable "
                "in this transformers version."
            )
        model_class = AutoModelForMultimodalLM
    else:
        model_class = AutoModelForCausalLM
    return model_class.from_pretrained(model_path, trust_remote_code=True, **kwargs)


def build_tokenizer(cfg: TrainConfig) -> AutoTokenizer:
    config = AutoConfig.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if _is_multimodal(config):
        # Multimodal repos ship a processor; its inner tokenizer keeps the rest of the
        # (text-only) pipeline unchanged.
        processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "right"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True, padding_side="right"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Disable Qwen3 thinking mode globally so <think> blocks are never generated,
    # matching training data that has no thinking traces. Chat templates that dropped
    # the enable_thinking kwarg (e.g. newer model families) raise TypeError — retry
    # without it.
    _orig_apply = tokenizer.apply_chat_template
    def _apply_no_think(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        try:
            return _orig_apply(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return _orig_apply(*args, **kwargs)
    tokenizer.apply_chat_template = _apply_no_think
    return tokenizer


def build_processor(cfg: TrainConfig):
    """Full AutoProcessor for multimodal models, else None (text-only).

    Unlike ``build_tokenizer`` (which unwraps to the inner tokenizer for the
    text-only training pipeline), this keeps the whole processor so the image/video
    preprocessor configs can be saved alongside checkpoints — vLLM needs them to
    load a ``finetuning_type: full`` VL checkpoint directly.
    """
    config = AutoConfig.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if _is_multimodal(config):
        return AutoProcessor.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
    return None


def save_processor(processor, output_dir: str) -> None:
    """Write the processor and its sub-processors into ``output_dir``.

    ``processor.save_pretrained`` alone (transformers 5.x) only emits a combined
    ``processor_config.json``; the VL loader path used by vLLM specifically requires
    ``preprocessor_config.json`` (and ``video_preprocessor_config.json``), so the
    image/video sub-processors are saved explicitly too.
    """
    processor.save_pretrained(output_dir)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.save_pretrained(output_dir)
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        video_processor.save_pretrained(output_dir)


class ProcessorSaveCallback(TrainerCallback):
    """Persist the full processor into each ``checkpoint-N/`` dir on save.

    HF ``Trainer`` only saves ``processing_class`` (here the bare tokenizer), so the
    image/video preprocessor configs would otherwise be missing from the periodic
    checkpoints that best-model resolution points inference at.
    """

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if self.processor is None or not state.is_world_process_zero:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_processor(self.processor, ckpt)


# JSONL filenames written by data_prep.py / opera_data_prep.py inside an output_dir.
_TRAIN_FILE, _VAL_FILE = "sft_train.jsonl", "sft_val.jsonl"


def _load_source_split(source: str, split: str) -> Optional[Dataset]:
    """Load one split of a dataset source; None when that split does not exist.

    A source is a HF Hub dataset id, a local directory holding the JSONL splits written by
    the data_prep scripts, or a single local .jsonl file (train only).
    """
    path = Path(source)
    if path.is_dir():
        local = path / (_TRAIN_FILE if split == "train" else _VAL_FILE)
        if not local.exists() or local.stat().st_size == 0:
            return None
        return load_dataset("json", data_files=str(local), split="train")
    if path.is_file():
        # A bare file is the train split; it carries no validation counterpart.
        if split != "train":
            return None
        return load_dataset("json", data_files=str(path), split="train")
    if source.endswith((".json", ".jsonl")):
        raise FileNotFoundError(f"Dataset source {source!r} does not exist.")
    try:
        return load_dataset(source, split=split)
    except ValueError:
        # datasets raises ValueError for an unknown split (e.g. a hub dataset pushed with
        # val_ratio 0, which has no "validation").
        if split == "train":
            raise
        return None


def _concat(parts: list[Dataset]) -> Dataset:
    """Concatenate dataset parts, narrowing them to their shared columns first.

    OPeRA and synthetic records carry slightly different bookkeeping columns, so the
    intersection (which must include "messages") is what actually lines up.
    """
    parts = [p for p in parts if p is not None and len(p) > 0]
    if len(parts) == 1:
        return parts[0]
    common = set(parts[0].column_names)
    for p in parts[1:]:
        common &= set(p.column_names)
    if "messages" not in common:
        raise ValueError(
            "Dataset sources share no 'messages' column; columns were "
            + " | ".join(str(p.column_names) for p in parts)
        )
    # Keep the first part's column order so the merged dataset looks like a single source.
    ordered = [c for c in parts[0].column_names if c in common]
    aligned = [p.select_columns(ordered) for p in parts]
    features = aligned[0].features
    aligned = [p if p.features == features else p.cast(features) for p in aligned]
    return concatenate_datasets(aligned)


def _dataset_sources(cfg: TrainConfig) -> tuple[list[str], Optional[str]]:
    """Resolve (train sources, eval source) from the config.

    ``train_datasets`` wins when set; otherwise fall back to the single-source
    ``dataset_hub_id`` / ``train_data_path`` fields so existing configs keep working.
    """
    if cfg.train_datasets:
        sources = [cfg.train_datasets] if isinstance(cfg.train_datasets, str) else list(cfg.train_datasets)
        return sources, cfg.eval_dataset or sources[0]
    if cfg.dataset_hub_id:
        return [cfg.dataset_hub_id], cfg.eval_dataset or cfg.dataset_hub_id
    return [cfg.train_data_path], cfg.eval_dataset


def _load_datasets(
    cfg: TrainConfig,
    num_train_sessions: Optional[int] = None,
):
    """Load training data and an independently prepared validation split.

    Validation resolution is: explicit ``eval_dataset`` validation split, explicit
    ``val_data_path``, then the first training source's validation split. Training rows are
    never split inside the trainer. An explicitly configured ``eval_dataset`` is strict: it
    must provide a non-empty validation split rather than silently falling back to synthetic
    training data.
    """
    train_sources, eval_source = _dataset_sources(cfg)

    parts = []
    for source in train_sources:
        part = _load_source_split(source, "train")
        if part is None or len(part) == 0:
            raise ValueError(f"Dataset source {source!r} has no train rows.")
        logger.info("Loaded %d train rows from %s", len(part), source)
        parts.append(part)
    train = _concat(parts)
    if len(parts) > 1:
        # Interleave the sources so a mixed run doesn't see one source as a contiguous block.
        train = train.shuffle(seed=cfg.seed)
        logger.info("Mixed %d sources into %d train rows", len(parts), len(train))

    # This cap affects training only; validation always comes from an independent source.
    if num_train_sessions is not None and 0 < num_train_sessions < len(train):
        logger.info("Sampling first %d/%d train rows", num_train_sessions, len(train))
        train = train.select(range(num_train_sessions))

    eval_ds = None
    eval_label = None
    if cfg.eval_dataset:
        eval_ds = _load_source_split(cfg.eval_dataset, "validation")
        eval_label = cfg.eval_dataset
        if eval_ds is None or len(eval_ds) == 0:
            raise ValueError(
                f"Explicit eval_dataset {cfg.eval_dataset!r} has no non-empty "
                "'validation' split. Refusing to split training data for validation."
            )

    if eval_ds is None and cfg.val_data_path:
        val_path = Path(cfg.val_data_path)
        if val_path.exists() and val_path.stat().st_size > 0:
            eval_ds = load_dataset("json", data_files=str(val_path), split="train")
            eval_label = str(val_path)
        else:
            logger.warning(
                "Configured val_data_path %s is missing or empty; trying the training "
                "source's prepared validation split.",
                val_path,
            )

    if eval_ds is None and not cfg.eval_dataset and eval_source:
        # No explicit validation setting: use the conventional validation split/file from
        # the first training source when it exists.
        eval_ds = _load_source_split(eval_source, "validation")
        eval_label = eval_source

    if eval_ds is not None and len(eval_ds) == 0:
        logger.warning("Validation source %s is empty; training without evaluation.", eval_label)
        eval_ds = None

    if eval_ds is None:
        logger.warning(
            "No independently prepared validation split was found; using all %d rows for "
            "training and training without evaluation.",
            len(train),
        )
    else:
        logger.info("Loaded %d eval rows from %s", len(eval_ds), eval_label)
    return train, eval_ds


def _sft_config(cfg: TrainConfig, has_eval: bool, effective_max_steps: int) -> SFTConfig:
    kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=effective_max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,
        use_liger_kernel=cfg.use_liger_kernel,
        optim=cfg.optim,
        dataloader_num_workers=cfg.dataloader_num_workers,
        packing=False,
        max_length=cfg.max_length,
        assistant_only_loss=cfg.assistant_only_loss,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy if has_eval else "no",
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=cfg.load_best_model_at_end and has_eval,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        run_name=cfg.wandb_run_name,
        seed=cfg.seed,
    )
    supported = {item.name for item in fields(SFTConfig)}
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = cfg.warmup_ratio
    elif "warmup_steps" in supported:
        # transformers 5.15 removed warmup_ratio. Its replacement accepts a
        # float in [0, 1) as a ratio, so the configured value keeps its meaning.
        kwargs["warmup_steps"] = cfg.warmup_ratio
    else:
        raise RuntimeError(
            "The installed SFTConfig supports neither warmup_ratio nor warmup_steps."
        )
    return SFTConfig(**kwargs)


def _drop_unsupervised_examples(trainer, max_length: int) -> None:
    """Drop examples that have zero supervised tokens after truncation.

    With assistant_only_loss, an example whose assistant turns all fall past
    `max_length` (e.g. a long persona + huge HTML observation pushes them out)
    produces a loss over an empty label set -> NaN -> the run diverges. This is
    a real failure mode on OPeRA, where observations are enormous. We filter
    such examples (per dataset) before training. The proper fix is Method 2
    (one row per step) or a larger max_length — see OPERA_PIPELINE_REPORT.md.
    """
    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr, None)
        if ds is None:
            continue
        cols = ds.column_names or []
        # TRL strips "assistant_masks" from the dataset when use_liger_kernel=True
        # (it keeps only {input_ids, seq_lengths, labels}), so we key off the
        # already-built "labels" column, which survives and encodes the same
        # supervision (non -100 == supervised token). Fall back to
        # "assistant_masks" only on the non-Liger path where labels may be absent.
        if "labels" in cols:
            is_supervised = lambda ex: any(t != -100 for t in ex["labels"][:max_length])
        elif "assistant_masks" in cols:
            is_supervised = lambda ex: sum(ex["assistant_masks"][:max_length]) > 0
        else:
            continue
        n0 = len(ds)
        kept = ds.filter(is_supervised)
        dropped = n0 - len(kept)
        if dropped:
            logger.warning(
                "Dropped %d/%d %s examples with 0 supervised tokens after truncation to %d "
                "(assistant turns fall past max_length). Use Method 2 or a larger max_length.",
                dropped, n0, attr, max_length,
            )
        if len(kept) == 0:
            raise RuntimeError(
                f"All {attr} examples lost their assistant tokens after truncation to "
                f"{max_length}. Increase max_length or switch to Method 2."
            )
        setattr(trainer, attr, kept)


def _assert_supervised_tokens(trainer) -> None:
    """Fail fast if loss masking leaves zero supervised tokens in any example.

    Catches the silent failure modes of assistant-only loss — a chat template
    without `{% generation %}` markers, or truncation that drops every assistant
    turn — before we waste a training run learning from nothing (or crashing DDP
    on an empty-label microbatch). Scans the *entire* train dataset (not a
    sample); it runs once before training, off the hot path, so a rare bad
    window buried deep in a large dataset can't slip through.
    """
    ds = trainer.train_dataset
    if ds is None or len(ds) == 0:
        return
    max_length = getattr(trainer.args, "max_length", None)
    sl = slice(None, max_length)
    use_labels = "labels" in (ds.column_names or [])

    zero_idx = []
    min_c = None
    max_c = 0
    for i in range(len(ds)):
        if use_labels:
            # "labels" survives TRL's Liger column-selection; count non -100 within
            # the truncation window the model actually sees (keep_start default).
            c = sum(1 for t in ds[i]["labels"][sl] if t != -100)
        else:
            batch = trainer.data_collator([ds[i]])
            c = int((batch["labels"] != -100).sum())
        if c == 0:
            zero_idx.append(i)
        min_c = c if min_c is None else min(min_c, c)
        max_c = max(max_c, c)

    if zero_idx:
        shown = zero_idx[:20]
        raise RuntimeError(
            f"{len(zero_idx)} of {len(ds)} train examples have 0 supervised tokens "
            f"after truncation (indices {shown}{' ...' if len(zero_idx) > len(shown) else ''}). "
            "Check assistant_only_loss / chat-template `{% generation %}` markers / max_length."
        )
    logger.info(
        "Supervised-token check OK: all %d train examples have >0 label tokens "
        "(min=%d, max=%d per example).",
        len(ds), min_c, max_c,
    )


def train(
    cfg: TrainConfig,
    max_steps: int = -1,
    check_only: bool = False,
    num_train_sessions: Optional[int] = None,
) -> None:
    set_seed(cfg.seed)
    tokenizer = build_tokenizer(cfg)
    processor = build_processor(cfg)
    train_ds, eval_ds = _load_datasets(
        cfg,
        num_train_sessions=num_train_sessions,
    )

    if cfg.finetuning_type == "lora":
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    elif cfg.finetuning_type == "full":
        peft_config = None
    else:
        raise ValueError(
            f"finetuning_type must be 'lora' or 'full', got {cfg.finetuning_type!r}"
        )
    sft_args = _sft_config(
        cfg,
        has_eval=eval_ds is not None,
        effective_max_steps=max_steps if max_steps > 0 else cfg.max_steps,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # DeepSpeed Zero-3 manages device placement itself; passing device_map raises an error
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    if use_deepspeed:
        device_map = None
    elif local_rank >= 0:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    # 4-bit QLoRA quantization only applies to LoRA; full fine-tuning trains the
    # (un-quantized) weights directly and loads them in bf16.
    quant_config = _bnb_config(cfg) if cfg.finetuning_type == "lora" else None
    if cfg.finetuning_type == "full" and cfg.use_4bit:
        logger.warning(
            "Full fine-tuning is incompatible with 4-bit quantization; disabling it."
        )

    logger.info("Initializing weights from %s", cfg.model_name_or_path)
    model = build_model(
        cfg.model_name_or_path,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if quant_config is None else None,
        device_map=device_map,
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True,
    )

    trainer = SFTTrainerWithGenEval(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        gen_eval_data=list(eval_ds) if eval_ds is not None else [],
        num_eval_generations=cfg.num_eval_generations,
        eval_max_new_tokens=cfg.eval_max_new_tokens,
        eval_gen_batch_size=cfg.eval_gen_batch_size,
        eval_temperature=cfg.eval_temperature,
    )

    if processor is not None:
        trainer.add_callback(ProcessorSaveCallback(processor))

    if cfg.early_stopping and eval_ds is not None:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        )

    if cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    _drop_unsupervised_examples(trainer, cfg.max_length)
    _assert_supervised_tokens(trainer)
    if check_only:
        logger.info("check_only=True — masking validated, skipping training.")
        return
    if eval_ds is not None:
        logger.info("Running baseline eval on the base model before training ...")
        trainer.evaluate()
    logger.info("Starting training ...")
    trainer.train()
    logger.info("Saving model to %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    # Also persist trainer_state.json at output_dir so eval can read best_model_checkpoint
    # directly (save_model alone does not write it; it only lives in checkpoint-*/ subdirs).
    trainer.save_state()
    tokenizer.save_pretrained(cfg.output_dir)
    # VL models need the image/video preprocessor configs to load the checkpoint dir
    # directly (full fine-tuning); the bare tokenizer above doesn't write them.
    if processor is not None and trainer.is_world_process_zero():
        save_processor(processor, cfg.output_dir)


def run_inference(
    cfg: TrainConfig,
    adapter_path: str,
    messages: list[dict],
    max_new_tokens: int = 512,
) -> str:
    tokenizer = build_tokenizer(cfg)
    if cfg.finetuning_type == "full":
        model = build_model(
            adapter_path,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    else:
        base = build_model(
            cfg.model_name_or_path,
            quantization_config=_bnb_config(cfg),
            dtype=getattr(torch, cfg.bnb_4bit_compute_dtype) if not cfg.use_4bit else None,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, adapter_path).eval()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT trainer for buyer simulation.")
    parser.add_argument("--config", default="user_model/sft/config.yaml")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override num_train_epochs with a fixed step count (smoke test).")
    parser.add_argument("--num_train_sessions", type=int, default=None,
                        help="Cap the train dataset to the first N rows. Omit to use all data.")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--check_only", action="store_true",
                        help="Build the trainer, assert loss masking supervises >0 tokens, then exit.")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", default=None,
                        help="Override output_dir from the config file.")
    parser.add_argument("--train_datasets", nargs="+", default=None,
                        help="Override train_datasets: one or more sources (HF Hub id, local "
                             "dir of sft_*.jsonl, or a .jsonl file) trained on concatenated.")
    parser.add_argument("--eval_dataset", default=None,
                        help="Override eval_dataset: the source whose validation split drives "
                             "eval / early stopping. Defaults to the first --train_datasets entry.")
    parser.add_argument("--model_name_or_path", default=None,
                        help="Override model_name_or_path — e.g. a previous run's output_dir, to "
                             "continue fine-tuning from that checkpoint (sequential training). "
                             "Full fine-tuning only: a LoRA checkpoint dir holds an adapter, not "
                             "base weights, so pass --adapter_path for that instead.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning_rate from the config file.")
    parser.add_argument("--wandb_run_name", default=None,
                        help="Override wandb_run_name from the config file.")
    return parser.parse_args()


def _demo_inference(cfg: TrainConfig, adapter_path: str) -> None:
    train_sources, eval_source = _dataset_sources(cfg)
    source = eval_source or train_sources[0]
    demo_ds = _load_source_split(source, "validation")
    if demo_ds is None:
        logger.info("No validation split in %s; using its first train record for demo inference.", source)
        demo_ds = _load_source_split(source, "train")
    if demo_ds is None or len(demo_ds) == 0:
        raise ValueError(f"Dataset source {source!r} has no rows for demo inference.")
    first = demo_ds[0]
    demo: list[dict] = []
    for msg in first["messages"]:
        demo.append(msg)
        if msg["role"] == "user":
            break
    logger.info("Inference on session %s", first.get("session_id"))
    print("=== Model output ===")
    print(run_inference(cfg, adapter_path, demo))


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    for key in ("output_dir", "train_datasets", "eval_dataset",
                "model_name_or_path", "learning_rate", "wandb_run_name"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    if args.inference_only:
        if not args.adapter_path:
            raise ValueError("--adapter_path is required with --inference_only")
        _demo_inference(cfg, args.adapter_path)
    else:
        train(
            cfg,
            max_steps=args.max_steps,
            check_only=args.check_only,
            num_train_sessions=args.num_train_sessions,
        )
