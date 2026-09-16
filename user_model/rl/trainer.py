"""Offline turn-level RL trainer for the Qwen3.5 next-action model.

This path mirrors ``user_model.inference``: one training example contains every
message before an assistant turn, and the sampled completion is only that next
action.  No browser or live environment is used during training.

Quick start::

  accelerate launch --config_file user_model/rl/accelerate_config.yaml \
    -m user_model.rl.trainer \
    --config user_model/rl/config.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

from user_model.rl.data_prep import build_turn_datasets
from user_model.rl.rewards import (
    TurnFormulaRewardConfig,
    build_formula_rewards,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class TurnGRPOTrainConfig:
    # Original conversational SFT JSONL (it is flattened in memory).
    train_data_path: str = "data/rl/train.jsonl"
    val_data_path: Optional[str] = None
    dataset_hub_id: Optional[str] = None

    # For a full SFT checkpoint, point model_name_or_path at that checkpoint.
    # For an SFT LoRA, keep the base here and set sft_adapter_path. The mode below
    # controls GRPO itself: attach/continue LoRA, or update all model parameters.
    model_name_or_path: str = "Qwen/Qwen3.5-9B"
    sft_adapter_path: Optional[str] = None
    finetuning_type: str = "lora"  # lora | full
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_double_quant: bool = True
    attn_implementation: str = "sdpa"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Number of responses in each prompt group. This is distinct from TRL's
    # generation batch, which can contain several prompt groups at once.
    num_generations: int = 8
    # Global number of responses generated together. When unset, TRL derives
    # this from per_device_train_batch_size * world_size *
    # gradient_accumulation_steps, which can make generation OOM when gradient
    # accumulation is increased. Set it explicitly to decouple rollout memory
    # from the effective optimizer batch size.
    generation_batch_size: Optional[int] = None
    # TRL 1.7 removed GRPOConfig.max_prompt_length. This is enforced directly on
    # conversational prompts before they reach GRPOTrainer.
    prompt_max_tokens: int = 16384
    # Cache the tokenizer-measured, truncated dataset so repeated launches do
    # not scan every long HTML prompt again. The default directory is inside
    # output_dir; set prompt_cache_dir to share it across output directories.
    cache_truncated_prompts: bool = True
    prompt_cache_dir: Optional[str] = None
    # Backward-compatible alias for older configs; load_config maps it to the
    # field above and it is never passed to GRPOConfig.
    max_prompt_length: Optional[int] = None
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.04
    use_vllm: bool = False
    # Server mode keeps rollout generation on CUDA devices that are disjoint
    # from the training workers. TRL synchronizes the current policy weights to
    # the server before the first rollout after every optimizer step.
    vllm_mode: str = "colocate"  # colocate | server
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 600.0
    vllm_group_port: int = 51216

    # Selects how R_format, R_action, and R_target become policy advantages.
    advantage_formula: str = "constraint-aware"  # gdpo | constraint-aware

    # Passed directly to TurnFormulaRewardConfig.
    reward: dict[str, Any] = field(default_factory=dict)

    output_dir: str = "user_model/rl/checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: Optional[bool] = True
    optim: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 4
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: Optional[int] = 3
    log_completions: bool = True
    # TRL uses log_completions for both the W&B table and a Rich console table.
    # Keep those independently controllable so W&B does not require noisy stdout.
    print_completion_tables: bool = False
    # Print every generated response with the composite reward and the exact
    # normalized advantage that TRL uses for the policy update.
    log_group_diagnostics: bool = False
    report_to: str = "none"
    wandb_project: str = "buyer-sim-gen"
    run_name: Optional[str] = None
    seed: int = 42
    max_steps: int = -1


def load_config(path: str) -> TurnGRPOTrainConfig:
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = TurnGRPOTrainConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logger.warning("Unknown config key: %s", key)
    if cfg.max_prompt_length is not None:
        if "prompt_max_tokens" in raw:
            logger.warning(
                "Both prompt_max_tokens and deprecated max_prompt_length are set; "
                "using prompt_max_tokens=%d",
                cfg.prompt_max_tokens,
            )
        else:
            cfg.prompt_max_tokens = cfg.max_prompt_length
            logger.warning(
                "max_prompt_length is deprecated and unsupported by TRL 1.7; "
                "enforcing the same value through prompt_max_tokens=%d",
                cfg.prompt_max_tokens,
            )
    if cfg.prompt_max_tokens <= 0:
        raise ValueError("prompt_max_tokens must be positive")
    if cfg.vllm_mode not in {"colocate", "server"}:
        raise ValueError("vllm_mode must be 'colocate' or 'server'")
    if not 1 <= cfg.vllm_server_port <= 65535:
        raise ValueError("vllm_server_port must be between 1 and 65535")
    if not 1 <= cfg.vllm_group_port <= 65535:
        raise ValueError("vllm_group_port must be between 1 and 65535")
    if cfg.vllm_server_timeout <= 0:
        raise ValueError("vllm_server_timeout must be positive")
    if cfg.advantage_formula not in {"gdpo", "constraint-aware"}:
        raise ValueError(
            "advantage_formula must be either 'gdpo' or 'constraint-aware'"
        )
    if not isinstance(cfg.reward, dict):
        raise ValueError("reward must be a YAML mapping")
    return cfg


def build_reward_config(cfg: TurnGRPOTrainConfig) -> TurnFormulaRewardConfig:
    try:
        reward_cfg = TurnFormulaRewardConfig(**cfg.reward)
    except TypeError as exc:
        raise ValueError(f"Invalid reward configuration: {exc}") from exc
    if cfg.advantage_formula == "constraint-aware":
        reward_cfg.validate()
    return reward_cfg


def _bnb_config(cfg: TurnGRPOTrainConfig):
    if not cfg.use_4bit or cfg.finetuning_type == "full":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_double_quant,
    )


_MULTIMODAL_ARCH_KEYWORDS = ("VL", "Vision", "ImageText", "Omni", "Multimodal")


def _is_multimodal(config) -> bool:
    """Match the Qwen3.5 architecture detection used by the SFT trainer."""
    if getattr(config, "vision_config", None) is not None:
        return True
    architectures = getattr(config, "architectures", None) or []
    return any(
        any(keyword in architecture for keyword in _MULTIMODAL_ARCH_KEYWORDS)
        for architecture in architectures
    )


def build_processing_classes(cfg: TurnGRPOTrainConfig):
    """Return ``(tokenizer, full_processor_or_none)`` for text-only GRPO.

    Qwen3.5 is a unified multimodal model, so its repository/checkpoints must be
    opened with ``AutoProcessor``. The GRPO examples themselves contain text-only
    chat messages, therefore the trainer receives the processor's inner tokenizer,
    exactly as in ``user_model.sft.trainer``. The full processor is retained for saving.
    """
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if _is_multimodal(config):
        processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True, padding_side="left"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Match SFT and inference: Qwen must emit the flat JSON directly, without a
    # hidden/visible <think> block changing the format reward.
    original_apply = tokenizer.apply_chat_template

    def apply_without_thinking(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        try:
            return original_apply(*args, **kwargs)
        except TypeError:
            # Some Qwen3.5 chat-template revisions no longer expose this kwarg.
            kwargs.pop("enable_thinking", None)
            return original_apply(*args, **kwargs)

    tokenizer.apply_chat_template = apply_without_thinking
    return tokenizer, processor


def _rendered_prompt_tokens(tokenizer, messages: list[dict[str, Any]]) -> int:
    """Count the exact chat-template tokens GRPO generation will receive."""
    # Qwen3.5 may return a structured/multimodal object from
    # apply_chat_template(tokenize=True). Rendering first and then tokenizing is
    # the same robust path used by user_model.sft.trainer._left_truncate_messages.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template rendered an empty or non-text prompt")
    encoded = tokenizer(rendered, add_special_tokens=False)
    # transformers.BatchEncoding is a UserDict/Mapping rather than a built-in
    # dict. Calling len() on it counts fields (usually 2), not tokens.
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids", [])
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    if encoded and isinstance(encoded[0], (list, tuple)):
        return len(encoded[0])
    return len(encoded)


# Increment this whenever prompt rendering or truncation behavior changes in a
# way that should invalidate previously materialized Arrow caches.
_PROMPT_CAP_CACHE_VERSION = 1
_PROMPT_CAP_STAT_COLUMNS = (
    "__prompt_cap_before_tokens",
    "__prompt_cap_after_tokens",
    "__prompt_cap_dropped_messages",
)


def _prompt_cap_cache_key(dataset, tokenizer, max_tokens: int) -> str:
    """Fingerprint every input that can affect the truncated prompt dataset."""
    digest = hashlib.sha256()

    def update(label: str, value: Any) -> None:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\0")

    update("cache_version", _PROMPT_CAP_CACHE_VERSION)
    update("dataset_fingerprint", getattr(dataset, "_fingerprint", None))
    update("max_tokens", max_tokens)
    update("tokenizer_class", type(tokenizer).__qualname__)
    update("tokenizer_name_or_path", getattr(tokenizer, "name_or_path", None))
    update("chat_template", getattr(tokenizer, "chat_template", None))
    update("special_tokens_map", getattr(tokenizer, "special_tokens_map", None))
    try:
        transformers_version = version("transformers")
    except Exception:  # pragma: no cover - transformers exists in real training
        transformers_version = None
    update("transformers_version", transformers_version)

    # Fast tokenizers expose their complete normalization, pre-tokenization,
    # vocabulary, and post-processing pipeline as stable JSON. This prevents a
    # stale hit when a tokenizer is replaced at the same checkpoint path.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    update("tokenizer_backend", backend_json)
    return digest.hexdigest()[:24]


def truncate_prompt_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Drop oldest complete exchanges until a conversational prompt fits.

    Returns ``(messages, original_tokens, final_tokens, dropped_messages)``.
    System messages and the latest user observation are always retained. Candidate
    cut points begin at user messages, preventing an orphan assistant answer from
    becoming the first history item.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt must be a non-empty message list")

    original_tokens = _rendered_prompt_tokens(tokenizer, messages)
    if original_tokens <= max_tokens:
        return list(messages), original_tokens, original_tokens, 0

    system = [message for message in messages if message.get("role") == "system"]
    history = [message for message in messages if message.get("role") != "system"]
    if not history or history[-1].get("role") != "user":
        raise ValueError("prompt must end with the current user observation")

    # Each valid start preserves a complete suffix beginning with a user message.
    starts = [
        idx for idx, message in enumerate(history)
        if message.get("role") == "user"
    ]
    for start in starts:
        if start == 0:  # the untruncated prompt was already measured above
            continue
        candidate = system + history[start:]
        candidate_tokens = _rendered_prompt_tokens(tokenizer, candidate)
        if candidate_tokens <= max_tokens:
            return candidate, original_tokens, candidate_tokens, start

    minimal = system + [history[-1]]
    minimal_tokens = _rendered_prompt_tokens(tokenizer, minimal)
    if minimal_tokens > max_tokens:
        raise ValueError(
            "system message plus current observation requires "
            f"{minimal_tokens} tokens, exceeding prompt_max_tokens={max_tokens}; "
            "increase the cap or simplify the current HTML observation"
        )
    # ``starts`` normally includes the final user message; retain this fallback
    # for malformed histories with unusual role values between turns.
    return minimal, original_tokens, minimal_tokens, len(history) - 1


def truncate_dataset_prompts(
    dataset,
    tokenizer,
    max_tokens: int,
    split_name: str,
    cache_dir: Optional[str] = None,
):
    """Apply the hard prompt cap, optionally using a persistent DDP-safe cache."""
    if dataset is None:
        return None
    collisions = set(dataset.column_names).intersection(_PROMPT_CAP_STAT_COLUMNS)
    if collisions:
        raise ValueError(f"Dataset contains reserved prompt-cache columns: {sorted(collisions)}")

    def truncate_row(row, index):
        try:
            prompt, before, after, dropped = truncate_prompt_messages(
                tokenizer, row["prompt"], max_tokens
            )
        except ValueError as exc:
            identity = row.get("session_id", "")
            turn = row.get("turn_idx", index)
            raise ValueError(
                f"Cannot truncate {split_name} prompt {identity}#turn-{turn}: {exc}"
            ) from exc
        return {
            "prompt": prompt,
            _PROMPT_CAP_STAT_COLUMNS[0]: before,
            _PROMPT_CAP_STAT_COLUMNS[1]: after,
            _PROMPT_CAP_STAT_COLUMNS[2]: dropped,
        }

    map_kwargs = {
        "with_indices": True,
        "desc": f"Enforcing {max_tokens}-token {split_name} prompt cap",
    }
    distributed_state = None
    cache_path = None
    cache_was_present = False
    if cache_dir:
        from accelerate import PartialState

        distributed_state = PartialState()
        cache_root = Path(cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = _prompt_cap_cache_key(dataset, tokenizer, max_tokens)
        safe_split = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in split_name
        )
        cache_path = cache_root / f"{safe_split}-{cache_key}.arrow"
        cache_was_present = cache_path.exists()
        map_kwargs.update(
            cache_file_name=str(cache_path),
            load_from_cache_file=True,
        )

        # Dataset.map's cache file is shared. Let global rank 0 finish its
        # atomic write before the remaining ranks attempt to load it.
        with distributed_state.main_process_first():
            result = dataset.map(truncate_row, **map_kwargs)
    else:
        map_kwargs["load_from_cache_file"] = False
        result = dataset.map(truncate_row, **map_kwargs)

    before_values = result[_PROMPT_CAP_STAT_COLUMNS[0]]
    after_values = result[_PROMPT_CAP_STAT_COLUMNS[1]]
    dropped_values = result[_PROMPT_CAP_STAT_COLUMNS[2]]
    stats = {
        "rows": len(result),
        "truncated": sum(value > 0 for value in dropped_values),
        "dropped": sum(dropped_values),
        "before_max": max(before_values, default=0),
        "after_max": max(after_values, default=0),
    }
    result = result.remove_columns(list(_PROMPT_CAP_STAT_COLUMNS))

    if distributed_state is None or distributed_state.is_main_process:
        if cache_path is not None:
            logger.info(
                "Prompt-cap cache %s: %s",
                "hit" if cache_was_present else "written",
                cache_path,
            )
        logger.info(
            "%s prompt cap: %d/%d rows truncated, %d old messages dropped; "
            "maximum %d -> %d tokens",
            split_name,
            stats["truncated"],
            stats["rows"],
            stats["dropped"],
            stats["before_max"],
            stats["after_max"],
        )
    return result


def save_processor(processor, output_dir: str) -> None:
    """Persist Qwen3.5 processor plus image/video preprocessor metadata."""
    processor.save_pretrained(output_dir)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.save_pretrained(output_dir)
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        video_processor.save_pretrained(output_dir)


def _processor_save_callback(processor):
    """Build a Trainer callback without importing transformers at module import."""
    from transformers import TrainerCallback

    class ProcessorSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            save_processor(
                processor,
                os.path.join(args.output_dir, f"checkpoint-{state.global_step}"),
            )

    return ProcessorSaveCallback()


def _device_map():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    uses_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    if uses_deepspeed:
        return None
    if local_rank >= 0:
        return {"": local_rank}
    return "auto"


def _load_model(cfg: TurnGRPOTrainConfig):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        AutoModelForMultimodalLM = None

    if cfg.finetuning_type not in {"lora", "full"}:
        raise ValueError("finetuning_type must be 'lora' or 'full'")
    if cfg.sft_adapter_path and cfg.finetuning_type != "lora":
        raise ValueError("sft_adapter_path is only valid with finetuning_type: lora")
    if cfg.finetuning_type == "full" and cfg.use_4bit:
        logger.warning("Full fine-tuning cannot use 4-bit weights; loading bf16 instead")

    quantization = _bnb_config(cfg)
    model_config = AutoConfig.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )
    if _is_multimodal(model_config):
        if AutoModelForMultimodalLM is None:
            raise RuntimeError(
                f"{cfg.model_name_or_path} is multimodal but "
                "AutoModelForMultimodalLM is unavailable; install transformers 5.x"
            )
        model_class = AutoModelForMultimodalLM
    else:
        model_class = AutoModelForCausalLM
    logger.info("Loading %s with %s", cfg.model_name_or_path, model_class.__name__)
    model = model_class.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=quantization,
        dtype=torch.bfloat16 if quantization is None else None,
        trust_remote_code=True,
        device_map=_device_map(),
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True,
    )
    if cfg.sft_adapter_path:
        logger.info("Continuing the trainable SFT adapter at %s", cfg.sft_adapter_path)
        model = PeftModel.from_pretrained(
            model, cfg.sft_adapter_path, is_trainable=True
        )
    return model


def _grpo_config(cfg: TurnGRPOTrainConfig):
    from trl import GRPOConfig

    kwargs = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "bf16": cfg.bf16,
        "fp16": False,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": cfg.ddp_find_unused_parameters,
        "optim": cfg.optim,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "num_generations": cfg.num_generations,
        "generation_batch_size": cfg.generation_batch_size,
        "max_completion_length": cfg.max_completion_length,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "beta": cfg.beta,
        "multi_objective_aggregation": (
            "normalize_then_sum"
            if cfg.advantage_formula == "gdpo"
            else "sum_then_normalize"
        ),
        # GDPO receives three separately normalized objectives. Constraint-aware
        # receives one already-weighted composite reward.
        "reward_weights": (
            [1.0, 1.0, 1.0]
            if cfg.advantage_formula == "gdpo"
            else [1.0]
        ),
        "use_vllm": cfg.use_vllm,
        "vllm_mode": cfg.vllm_mode,
        "vllm_server_base_url": cfg.vllm_server_base_url,
        "vllm_server_host": cfg.vllm_server_host,
        "vllm_server_port": cfg.vllm_server_port,
        "vllm_server_timeout": cfg.vllm_server_timeout,
        "vllm_group_port": cfg.vllm_group_port,
        "remove_unused_columns": False,
        "log_completions": cfg.log_completions,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
    }
    supported = {item.name for item in dataclasses.fields(GRPOConfig)}
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = cfg.warmup_ratio
    elif "warmup_steps" in supported:
        # transformers 5.15 represents ratios as fractional warmup_steps.
        kwargs["warmup_steps"] = cfg.warmup_ratio
    else:
        logger.warning(
            "GRPOConfig supports neither warmup_ratio nor warmup_steps; "
            "warmup is disabled."
        )
    if (
        cfg.advantage_formula == "gdpo"
        and "multi_objective_aggregation" not in supported
    ):
        raise RuntimeError(
            "The installed TRL does not support native GDPO. Install TRL >= 1.7.0 "
            "with GRPOConfig.multi_objective_aggregation."
        )
    dropped = sorted(key for key in kwargs if key not in supported)
    if dropped:
        logger.warning("TRL %s ignores GRPOConfig args: %s", version("trl"), dropped)
    return GRPOConfig(**{key: value for key, value in kwargs.items() if key in supported})


class _TurnGroupDiagnosticsMixin:
    """Log TRL's final rewards and advantages in prompt groups.

    The reward callable runs before distributed gathering and before advantage
    normalization, so it cannot reliably display a complete group.  This hook
    runs after ``GRPOTrainer`` has gathered rewards and computed the advantages
    that will actually be used by the loss.
    """

    log_group_diagnostics = False
    print_completion_tables = False

    def _latest_aggregated_rewards(self, response_count: int) -> list[float]:
        """Combine raw logged objectives for diagnostics only.

        GDPO's actual learning signal remains TRL's separately normalized and
        batch-normalized advantage. This sum is only a readable raw-reward log.
        """

        weights = getattr(self, "reward_weights", None)
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
        if weights is None:
            weights = [1.0] * len(self.reward_func_names)
        columns = [
            list(self._logs["rewards"][name])[-response_count:]
            for name in self.reward_func_names
        ]
        if not columns or any(len(column) != response_count for column in columns):
            return []
        return [
            sum(
                float(weight) * float(column[idx])
                for weight, column in zip(weights, columns)
            )
            for idx in range(response_count)
        ]

    def log(self, logs, start_time=None):
        """Retain W&B completion tables while optionally suppressing Rich stdout.

        TRL 1.7 controls both outputs with ``log_completions``. Its console
        branch is guarded by a module-level ``is_rich_available`` function, so
        temporarily disabling only that guard leaves parquet and W&B table
        logging unchanged.
        """
        base_log = super().log
        if self.print_completion_tables or not self.log_completions:
            return base_log(logs, start_time)

        method_globals = getattr(
            getattr(base_log, "__func__", None), "__globals__", {}
        )
        rich_available = method_globals.get("is_rich_available")
        if rich_available is None:
            return base_log(logs, start_time)
        method_globals["is_rich_available"] = lambda: False
        try:
            return base_log(logs, start_time)
        finally:
            method_globals["is_rich_available"] = rich_available

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        self._record_rollout_metrics(inputs)
        if self.log_group_diagnostics and self.accelerator.is_main_process:
            self._log_latest_turn_groups(inputs)
        return result

    def _record_rollout_metrics(self, inputs) -> None:
        """Log paper-style rollout metrics with an explicit W&B step axis."""
        mode = "train" if self.model.training else "eval"
        response_count = len(inputs) * self.accelerator.num_processes
        rewards = self._latest_aggregated_rewards(response_count)
        advantages = list(self._logs["advantages"])[-response_count:]
        rewards = [float(value) for value in rewards if math.isfinite(float(value))]
        advantages = [
            float(value) for value in advantages if math.isfinite(float(value))
        ]
        extras = self._logs.get("extra", {})

        def latest_extra(name: str) -> list[Any]:
            values = list(extras.get(name, []))[-response_count:]
            return values if len(values) == response_count else []

        format_valid = latest_extra("format_valid")
        action_match = latest_extra("action_match")
        target_applicable = latest_extra("target_applicable")
        target_match = latest_extra("target_match")
        target_score = latest_extra("target_score")
        ui_judge_reward = latest_extra("ui_judge_reward")
        reasoning_reward = latest_extra("reasoning_reward")
        judge_attempted = latest_extra("ui_judge_attempted")
        judge_failed = latest_extra("ui_judge_failed")
        zero_reward = latest_extra("zero_reward")

        def mean(values: list[Any]) -> float:
            return sum(float(value) for value in values) / len(values)

        metrics = {}
        if rewards:
            # Every group has the same num_generations, so the mean over all
            # responses is exactly the mean of the per-group reward means.
            metrics["rollout/rewards"] = sum(rewards) / len(rewards)
        if advantages:
            # Group-relative advantages are centered, making their signed mean
            # approximately zero. Mean absolute magnitude is the useful scalar
            # learning-signal curve; retain the signed mean as a sanity check.
            metrics["rollout/advantages"] = (
                sum(abs(value) for value in advantages) / len(advantages)
            )
            metrics["rollout/advantages_signed_mean"] = (
                sum(advantages) / len(advantages)
            )
        if format_valid:
            metrics["rollout/format_valid_rate"] = mean(format_valid)
        if action_match:
            metrics["rollout/action_match_rate"] = mean(action_match)
        if target_match and target_applicable:
            applicable_matches = [
                matched
                for matched, applicable in zip(target_match, target_applicable)
                if applicable
            ]
            if applicable_matches:
                metrics["rollout/target_match_rate"] = mean(applicable_matches)
        if target_score and target_applicable:
            applicable_scores = [
                score
                for score, applicable in zip(target_score, target_applicable)
                if applicable
            ]
            if applicable_scores:
                metrics["rollout/target_score"] = mean(applicable_scores)

        # UI and reasoning are only evaluated after the format and action gates.
        # Average them over evaluated responses instead of treating skipped
        # candidates as judge scores of zero.
        if ui_judge_reward and judge_attempted:
            evaluated_ui = [
                score
                for score, attempted in zip(ui_judge_reward, judge_attempted)
                if attempted
            ]
            if evaluated_ui:
                metrics["rollout/ui_judge_reward"] = mean(evaluated_ui)
        if reasoning_reward and format_valid and action_match:
            evaluated_reasoning = [
                score
                for score, valid, matched in zip(
                    reasoning_reward, format_valid, action_match
                )
                if valid and matched
            ]
            if evaluated_reasoning:
                metrics["rollout/reasoning_reward"] = mean(evaluated_reasoning)
        if judge_failed and judge_attempted:
            attempted_failures = [
                failed
                for failed, attempted in zip(judge_failed, judge_attempted)
                if attempted
            ]
            if attempted_failures:
                metrics["rollout/judge_failure_rate"] = mean(attempted_failures)
        if zero_reward:
            metrics["rollout/zero_reward_fraction"] = mean(zero_reward)
        if not metrics:
            return

        report_to = self.args.report_to
        uses_wandb = report_to == "wandb" or "wandb" in (report_to or [])
        if uses_wandb and self.accelerator.is_main_process:
            import wandb

            if wandb.run is not None:
                if not getattr(self, "_rollout_wandb_metrics_defined", False):
                    wandb.define_metric("rollout/step")
                    for name in (
                        "rollout/rewards",
                        "rollout/advantages",
                        "rollout/advantages_signed_mean",
                        "rollout/format_valid_rate",
                        "rollout/action_match_rate",
                        "rollout/target_match_rate",
                        "rollout/target_score",
                        "rollout/ui_judge_reward",
                        "rollout/reasoning_reward",
                        "rollout/judge_failure_rate",
                        "rollout/zero_reward_fraction",
                    ):
                        wandb.define_metric(name, step_metric="rollout/step")
                    self._rollout_wandb_metrics_defined = True
                # The rollout is sampled before the optimizer increments
                # global_step; label it with the update it will drive.
                wandb.log(
                    {"rollout/step": self.state.global_step + 1, **metrics}
                )
                return

        # Preserve the same metrics for non-W&B reporters and local logs.
        for name, value in metrics.items():
            self._metrics[mode][name].append(value)

    def _log_latest_turn_groups(self, inputs) -> None:
        mode = "train" if self.model.training else "eval"
        num_generations = (
            self.num_generations
            if mode == "train"
            else self.num_generations_eval
        )
        # Each process starts with len(inputs) responses. TRL gathers them in
        # process order before grouping and appending to _logs.
        response_count = len(inputs) * self.accelerator.num_processes
        prompts = list(self._logs["prompt"])[-response_count:]
        completions = list(self._logs["completion"])[-response_count:]
        advantages = list(self._logs["advantages"])[-response_count:]
        rewards = self._latest_aggregated_rewards(response_count)
        extras = self._logs.get("extra", {})
        session_ids = list(extras.get("session_id", []))[-response_count:]
        turn_indices = list(extras.get("turn_idx", []))[-response_count:]

        actual_count = min(len(completions), len(rewards), len(advantages))
        if actual_count == 0:
            logger.warning("No responses available for GRPO group diagnostics")
            return
        if actual_count % num_generations:
            logger.warning(
                "Cannot group %d logged responses into num_generations=%d",
                actual_count,
                num_generations,
            )
            return

        group_count = actual_count // num_generations
        logger.info(
            "GRPO %s generation batch: %d responses = %d groups x %d generations",
            mode,
            actual_count,
            group_count,
            num_generations,
        )
        for group_idx in range(group_count):
            start = group_idx * num_generations
            stop = start + num_generations
            labels = []
            if len(session_ids) == actual_count:
                labels.append(f"session_id={session_ids[start]!r}")
            if len(turn_indices) == actual_count:
                labels.append(f"turn_idx={turn_indices[start]!r}")
            label_text = f" ({', '.join(labels)})" if labels else ""
            logger.info(
                "GRPO group %d/%d%s",
                group_idx + 1,
                group_count,
                label_text,
            )
            if len(prompts) == actual_count and any(
                prompt != prompts[start] for prompt in prompts[start + 1 : stop]
            ):
                logger.warning(
                    "GRPO group %d contains responses from different prompts",
                    group_idx + 1,
                )
            if len(session_ids) == actual_count and len(
                set(session_ids[start:stop])
            ) > 1:
                logger.warning(
                    "GRPO group %d contains different session_ids: %r",
                    group_idx + 1,
                    session_ids[start:stop],
                )
            if len(turn_indices) == actual_count and len(
                set(turn_indices[start:stop])
            ) > 1:
                logger.warning(
                    "GRPO group %d contains different turn_idx values: %r",
                    group_idx + 1,
                    turn_indices[start:stop],
                )
            for response_idx in range(num_generations):
                idx = start + response_idx
                completion = json.dumps(
                    completions[idx], ensure_ascii=False, separators=(",", ":")
                )
                logger.info(
                    "  response %d/%d reward=%.10f advantage=%+.10f completion=%s",
                    response_idx + 1,
                    num_generations,
                    rewards[idx],
                    advantages[idx],
                    completion,
                )


def train(
    cfg: TurnGRPOTrainConfig,
    max_steps: int = -1,
    num_train_turns: Optional[int] = None,
) -> None:
    from peft import LoraConfig, TaskType
    from transformers import set_seed
    from trl import GRPOTrainer

    if max_steps > 0:
        cfg.max_steps = max_steps
    set_seed(cfg.seed)
    if cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    tokenizer, processor = build_processing_classes(cfg)
    train_ds, val_ds = build_turn_datasets(
        train_data_path=cfg.train_data_path,
        val_data_path=cfg.val_data_path,
        dataset_hub_id=cfg.dataset_hub_id,
    )
    if num_train_turns and 0 < num_train_turns < len(train_ds):
        logger.info("Using the first %d/%d training turns", num_train_turns, len(train_ds))
        train_ds = train_ds.select(range(num_train_turns))
    prompt_cache_dir = None
    if cfg.cache_truncated_prompts:
        prompt_cache_dir = cfg.prompt_cache_dir or os.path.join(
            cfg.output_dir, "prompt-cap-cache"
        )
    train_ds = truncate_dataset_prompts(
        train_ds,
        tokenizer,
        cfg.prompt_max_tokens,
        "train",
        cache_dir=prompt_cache_dir,
    )
    val_ds = truncate_dataset_prompts(
        val_ds,
        tokenizer,
        cfg.prompt_max_tokens,
        "validation",
        cache_dir=prompt_cache_dir,
    )

    reward_cfg = build_reward_config(cfg)
    logger.info("Turn reward config: %s", dataclasses.asdict(reward_cfg))
    rewards = build_formula_rewards(cfg.advantage_formula, reward_cfg)
    logger.info(
        "Advantage formula %s with rewards %s",
        cfg.advantage_formula,
        [reward.__name__ for reward in rewards],
    )
    model = _load_model(cfg)

    # Continue an existing SFT adapter in place. Otherwise attach a fresh LoRA,
    # or leave peft_config unset for full-parameter GRPO.
    peft_config = None
    if cfg.finetuning_type == "lora" and not cfg.sft_adapter_path:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    class TurnGRPOTrainer(_TurnGroupDiagnosticsMixin, GRPOTrainer):
        pass

    trainer = TurnGRPOTrainer(
        model=model,
        reward_funcs=rewards,
        args=_grpo_config(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.log_group_diagnostics = cfg.log_group_diagnostics
    trainer.print_completion_tables = cfg.print_completion_tables
    logger.info(
        "TRL generation batching: %d responses = per_device_batch_size=%d x "
        "steps_per_generation=%d x processes=%d; num_generations=%d gives %d groups",
        trainer.args.generation_batch_size,
        trainer.args.per_device_train_batch_size,
        trainer.args.steps_per_generation,
        trainer.accelerator.num_processes,
        trainer.num_generations,
        trainer.args.generation_batch_size // trainer.num_generations,
    )
    if processor is not None:
        trainer.add_callback(_processor_save_callback(processor))
    logger.info("Starting offline turn-level GRPO on %d prompts", len(train_ds))
    trainer.train()
    trainer.save_model(cfg.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(cfg.output_dir)
    if processor is not None and trainer.is_world_process_zero():
        save_processor(processor, cfg.output_dir)
    logger.info("Saved turn-level GRPO model to %s", cfg.output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline turn-level next-action GRPO")
    parser.add_argument("--config", default="user_model/rl/config.yaml")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_turns", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--advantage-formula",
        "--advantage_formula",
        choices=("gdpo", "constraint-aware"),
        dest="advantage_formula",
        help="Override the config's advantage calculation formula.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.advantage_formula:
        config.advantage_formula = args.advantage_formula
    train(config, max_steps=args.max_steps, num_train_turns=args.num_train_turns)
