"""Shared standalone inference and evaluation for buyer simulation models.

Loads the *best* checkpoint (lowest eval_loss) produced by ``user_model.sft.trainer``, runs
greedy generation over an eval dataset (HuggingFace Hub id or a local JSONL file), reports
action-prediction metrics, and dumps per-turn predictions.

Which checkpoint is loaded
--------------------------
Training sets ``load_best_model_at_end: true`` and re-saves the best adapter at the root of
``output_dir`` (via ``trainer.save_model``). The best ``checkpoint-N`` is recorded under
``best_model_checkpoint`` in a ``trainer_state.json`` — at ``<output_dir>`` only if
``trainer.save_state()`` ran, but always inside each ``checkpoint-*/`` subdir. This script
resolves the checkpoint by precedence: explicit ``--adapter_path`` > top-level
``trainer_state.json`` > the latest ``checkpoint-*/trainer_state.json`` > ``output_dir``
(unverified fallback). It logs its choice; pass ``--adapter_path`` to override.

Quick start
-----------
python -m user_model.inference --config user_model/sft/config.yaml \
    --adapter_path ckpt/sft --data data/sft/sft_eval.jsonl \
    --output_dir ckpt/sft --backend vllm --tp_size 2

# Baseline: eval the original base model before finetuning (no LoRA adapter).
python -m user_model.inference --config user_model/sft/config.yaml \
    --base_only --data data/sft/sft_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import f1_score as sklearn_f1
from tqdm import tqdm
from user_model.sft.eval_filters import filter_w0_records
from user_model.sft.trainer import (
    TrainConfig,
    _bnb_config,
    _dataset_sources,
    _iter_assistant_turns,
    _parse_method_target,
    build_model,
    build_tokenizer,
    load_config,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_DEFAULT_EVAL_FILE = "data/sft/sft_eval.jsonl"


def _resolve_best_path(best: str, output_dir: str) -> Optional[Path]:
    """Resolve a recorded ``best_model_checkpoint`` to an existing directory.

    ``best`` is stored relative to the training CWD (e.g.
    ``user_model/sweeps/.../checkpoint-40``), so it may not exist when eval runs from a
    different CWD. Fall back to reattaching the ``checkpoint-N`` basename to ``output_dir``.
    Returns the resolved Path if it exists, else None.
    """
    candidate = Path(best)
    if candidate.exists():
        return candidate
    reattached = Path(output_dir) / candidate.name
    if reattached.exists():
        return reattached
    return None


def resolve_checkpoint(output_dir: str, adapter_path: Optional[str] = None) -> str:
    """Return the adapter directory to load.

    Precedence: explicit --adapter_path > ``<output_dir>/trainer_state.json``
    best_model_checkpoint > best_model_checkpoint from the latest ``checkpoint-*/``
    subdir's trainer_state.json > output_dir.
    """
    if adapter_path:
        logger.info("Using explicit adapter path: %s", adapter_path)
        return adapter_path

    # Tier 2: top-level trainer_state.json (written only if trainer.save_state() ran).
    state_path = Path(output_dir) / "trainer_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            best = state.get("best_model_checkpoint")
            if best:
                resolved = _resolve_best_path(best, output_dir)
                if resolved is not None:
                    logger.info("Using best checkpoint from trainer_state.json: %s", resolved)
                    return str(resolved)
                logger.warning(
                    "best_model_checkpoint %s recorded but missing on disk; "
                    "trying checkpoint subdirs.", best,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s); trying checkpoint subdirs.", state_path, exc)

    # Tier 3: no usable top-level state — consult the per-checkpoint trainer_state.json
    # files. The one with the highest global_step carries the final best_model_checkpoint.
    latest_state, latest_step = None, -1
    for ckpt_dir in Path(output_dir).glob("checkpoint-*"):
        sub_state_path = ckpt_dir / "trainer_state.json"
        if not sub_state_path.exists():
            continue
        try:
            sub_state = json.loads(sub_state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s); skipping.", sub_state_path, exc)
            continue
        step = sub_state.get("global_step", -1)
        if step > latest_step:
            latest_state, latest_step = sub_state, step

    if latest_state is not None:
        best = latest_state.get("best_model_checkpoint")
        if best:
            resolved = _resolve_best_path(best, output_dir)
            if resolved is not None:
                logger.info(
                    "Using best checkpoint from checkpoint-%s/trainer_state.json: %s",
                    latest_step, resolved,
                )
                return str(resolved)
            logger.warning(
                "best_model_checkpoint %s recorded in checkpoint-%s but missing on disk; "
                "falling back to output_dir.", best, latest_step,
            )

    logger.info(
        "Using model at output_dir (unverified as best; no trainer_state.json resolved): %s",
        output_dir,
    )
    return output_dir


def _is_local_file(data_source: str) -> bool:
    return data_source.endswith((".jsonl", ".json")) or Path(data_source).exists()


def load_eval_data(
    cfg: TrainConfig, data_source: Optional[str] = None, split: str = "test"
) -> list[dict]:
    """Load eval records as a list of {"messages": [...]} dicts.

    Source resolution mirrors trainer._load_datasets:
      - --data pointing at a .jsonl/.json file -> local file
      - --data otherwise -> HuggingFace Hub id (using ``split``)
      - --data omitted -> the config's eval source (``eval_dataset``, else
        ``dataset_hub_id``, else the first ``train_datasets`` entry) read from the
        Hub at ``split``, else local sft_eval.jsonl (falling back to cfg.val_data_path)
    """
    # Fall back to whichever source the trainer would have evaluated on, so a config
    # written with train_datasets/eval_dataset works without an explicit --data.
    if not data_source:
        _, eval_source = _dataset_sources(cfg)
        if eval_source:
            local_eval = Path(eval_source) / "sft_eval.jsonl"
            # A local dir source keeps its own test split next to the train/val JSONL.
            data_source = str(local_eval) if local_eval.exists() else (
                eval_source if not Path(eval_source).exists() else None
            )

    if data_source:
        if _is_local_file(data_source):
            logger.info("Loading eval data from local file: %s", data_source)
            ds = load_dataset("json", data_files=data_source, split="train")
        else:
            logger.info("Loading eval data from HF Hub: %s [split=%s]", data_source, split)
            ds = load_dataset(data_source, split=split)
    else:
        local = _DEFAULT_EVAL_FILE if Path(_DEFAULT_EVAL_FILE).exists() else cfg.val_data_path
        if not local or not Path(local).exists():
            raise FileNotFoundError(
                f"No eval data source given and no local file found "
                f"(tried {_DEFAULT_EVAL_FILE!r} and {cfg.val_data_path!r})."
            )
        logger.info("Loading eval data from local file: %s", local)
        ds = load_dataset("json", data_files=local, split="train")

    return list(ds)


def load_model(cfg: TrainConfig, checkpoint_path: Optional[str] = None):
    """Load the model once, ready for generation.

    With a LoRA ``checkpoint_path`` the base model is wrapped with the adapter; with a
    ``finetuning_type: full`` checkpoint the checkpoint dir is a complete model and is loaded
    directly; without a checkpoint the original base model (before finetuning) is returned
    as-is for a baseline eval.
    """
    tokenizer = build_tokenizer(cfg)
    is_full_checkpoint = checkpoint_path is not None and cfg.finetuning_type == "full"
    if is_full_checkpoint:
        model = build_model(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        base = build_model(
            cfg.model_name_or_path,
            quantization_config=_bnb_config(cfg),
            dtype=getattr(torch, cfg.bnb_4bit_compute_dtype) if not cfg.use_4bit else None,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, checkpoint_path) if checkpoint_path else base
    model = model.eval()
    model.config.use_cache = True
    return model, tokenizer


def _flatten_turns(
    tokenizer, records: list[dict], max_length: Optional[int] = None, max_new_tokens: int = 0
) -> tuple[list[str], list[str], list[str]]:
    """Flatten records into parallel lists of (prompt_str, true_text, session_id) per turn.

    Prompts are pre-templated with the chat template (add_generation_prompt=True) so any
    backend can tokenize them directly without re-applying the template. When max_length is
    given, each prompt is left-truncated to fit within max_length - max_new_tokens tokens.
    """
    prompts: list[str] = []
    true_texts: list[str] = []
    session_ids: list[str] = []
    for record in records:
        session_id = record.get("session_id", "")
        for prompt_msgs, true_text in _iter_assistant_turns(
            record["messages"], tokenizer=tokenizer,
            max_length=max_length, max_new_tokens=max_new_tokens,
        ):
            if not true_text:
                continue
            prompts.append(tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ))
            true_texts.append(true_text)
            session_ids.append(session_id)
    return prompts, true_texts, session_ids


def _metrics_and_predictions(
    prompts: list[str], true_texts: list[str], session_ids: list[str], pred_texts: list[str]
) -> tuple[dict, list[dict]]:
    """Compute action-prediction metrics + per-turn prediction rows (backend-agnostic)."""
    pred_actions: list[str] = []
    true_actions: list[str] = []
    exact_hits: list[bool] = []
    predictions: list[dict] = []

    for pred_text, true_text, session_id in zip(pred_texts, true_texts, session_ids):
        pm, pt = _parse_method_target(pred_text)
        tm, tt = _parse_method_target(true_text)
        if not tm:
            continue
        hit = pm == tm and pt == tt
        pred_actions.append(pm)
        true_actions.append(tm)
        exact_hits.append(hit)
        predictions.append({
            "session_id": session_id,
            "true_text": true_text,
            "pred_text": pred_text,
            "true_action": tm,
            "pred_action": pm,
            "exact_match": hit,
        })

    n = len(true_actions)
    if n == 0:
        metrics = {
            "exact_match_acc": 0.0, "action_acc": 0.0,
            "action_f1": 0.0, "action_f1_weighted": 0.0, "num_turns": 0,
        }
    else:
        metrics = {
            "exact_match_acc": sum(exact_hits) / n,
            "action_acc": sum(p == t for p, t in zip(pred_actions, true_actions)) / n,
            "action_f1": sklearn_f1(true_actions, pred_actions, average="macro", zero_division=0),
            "action_f1_weighted": sklearn_f1(
                true_actions, pred_actions, average="weighted", zero_division=0
            ),
            "num_turns": n,
        }
    return metrics, predictions


def evaluate_vllm(
    cfg: TrainConfig,
    records: list[dict],
    checkpoint_path: Optional[str] = None,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    temperature: float = 0.0,
    tp_size: int = 1,
) -> tuple[dict, list[dict]]:
    """Generate every assistant turn with vLLM (continuous batching).

    For a LoRA checkpoint, serves the adapter directly via ``LoRARequest`` (no merge needed).
    For a ``finetuning_type: full`` checkpoint, loads the checkpoint dir as the model itself.
    Pass ``checkpoint_path=None`` for a base-model baseline. ``temperature=0.0`` is greedy.
    Metrics match ``evaluate``.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # vLLM's EngineCore worker is launched as a child process. If a CUDA context already
    # exists in this (parent) process when it forks, the child crashes trying to
    # re-initialize CUDA. Forcing 'spawn' avoids inheriting any such context.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if limit is not None:
        records = records[:limit]

    tokenizer = build_tokenizer(cfg)
    prompts, true_texts, session_ids = _flatten_turns(
        tokenizer, records, max_length=cfg.max_length, max_new_tokens=max_new_tokens
    )
    logger.info("vLLM eval over %d turns from %d records", len(prompts), len(records))

    is_full_checkpoint = checkpoint_path is not None and cfg.finetuning_type == "full"
    model_path = checkpoint_path if is_full_checkpoint else cfg.model_name_or_path
    enable_lora = checkpoint_path is not None and not is_full_checkpoint

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=cfg.lora_r,
        max_model_len=cfg.max_length,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
    )
    sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    lora_request = LoRARequest("adapter", 1, checkpoint_path) if enable_lora else None

    gen_start = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    gen_time = time.time() - gen_start
    pred_texts = [out.outputs[0].text for out in outputs]

    metrics, predictions = _metrics_and_predictions(prompts, true_texts, session_ids, pred_texts)
    metrics["generation_time_sec"] = round(gen_time, 3)
    return metrics, predictions


def evaluate_sglang(
    cfg: TrainConfig,
    records: list[dict],
    checkpoint_path: Optional[str] = None,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    temperature: float = 0.0,
    tp_size: int = 1,
) -> tuple[dict, list[dict]]:
    """Generate every assistant turn with SGLang's offline engine.

    RadixAttention prefix caching (default-on) reuses the KV cache across the long
    shared prefixes of turns within a session. For a LoRA checkpoint, serves the adapter
    directly. For a ``finetuning_type: full`` checkpoint, loads the checkpoint dir as the
    model itself. Pass ``checkpoint_path=None`` for a base-model baseline. ``temperature=0.0``
    is greedy. Metrics match ``evaluate``.
    """
    import sglang as sgl

    if limit is not None:
        records = records[:limit]

    tokenizer = build_tokenizer(cfg)
    prompts, true_texts, session_ids = _flatten_turns(
        tokenizer, records, max_length=cfg.max_length, max_new_tokens=max_new_tokens
    )
    logger.info("SGLang eval over %d turns from %d records", len(prompts), len(records))

    is_full_checkpoint = checkpoint_path is not None and cfg.finetuning_type == "full"
    model_path = checkpoint_path if is_full_checkpoint else cfg.model_name_or_path
    use_lora = checkpoint_path is not None and not is_full_checkpoint

    engine_kwargs = dict(
        model_path=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        context_length=cfg.max_length,
        tp_size=tp_size,
    )
    if use_lora:
        engine_kwargs.update(
            enable_lora=True,
            max_lora_rank=cfg.lora_r,
            max_loras_per_batch=2,  # one slot for the adapter, one for the base model
            lora_paths={"adapter": checkpoint_path},
        )

    engine = sgl.Engine(**engine_kwargs)
    try:
        sampling_params = {"temperature": temperature, "max_new_tokens": max_new_tokens}
        gen_kwargs = {}
        if use_lora:
            gen_kwargs["lora_path"] = ["adapter"] * len(prompts)
        gen_start = time.time()
        outputs = engine.generate(prompts, sampling_params, **gen_kwargs)
        gen_time = time.time() - gen_start
    finally:
        engine.shutdown()

    pred_texts = [out["text"] for out in outputs]

    metrics, predictions = _metrics_and_predictions(prompts, true_texts, session_ids, pred_texts)
    metrics["generation_time_sec"] = round(gen_time, 3)
    return metrics, predictions


def evaluate(
    model,
    tokenizer,
    records: list[dict],
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    temperature: float = 0.0,
    max_length: Optional[int] = None,
) -> tuple[dict, list[dict]]:
    """Generate each assistant turn and compute action-prediction metrics.

    ``temperature=0.0`` is greedy. Returns (metrics, predictions). Metrics match the
    training-time gen-eval: exact_match_acc, action_acc, action_f1 (macro), plus
    action_f1_weighted (support-weighted).
    """
    if limit is not None:
        records = records[:limit]

    # temperature <= 0 → greedy; > 0 → sampling at that temperature.
    gen_kwargs = (
        {"do_sample": True, "temperature": temperature}
        if temperature and temperature > 0
        else {"do_sample": False}
    )

    pred_actions: list[str] = []
    true_actions: list[str] = []
    exact_hits: list[bool] = []
    predictions: list[dict] = []
    gen_time = 0.0

    bar = tqdm(records, desc="Evaluating", unit="record")
    for record in bar:
        session_id = record.get("session_id", "")
        for prompt_msgs, true_text in _iter_assistant_turns(
            record["messages"], tokenizer=tokenizer,
            max_length=max_length, max_new_tokens=max_new_tokens,
        ):
            if not true_text:
                continue
            prompt = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_start = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs,
                )
            gen_time += time.time() - gen_start
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)

            pm, pt = _parse_method_target(pred_text)
            tm, tt = _parse_method_target(true_text)
            if not tm:
                continue
            hit = pm == tm and pt == tt
            pred_actions.append(pm)
            true_actions.append(tm)
            exact_hits.append(hit)
            predictions.append(
                {
                    "session_id": session_id,
                    "true_text": true_text,
                    "pred_text": pred_text,
                    "true_action": tm,
                    "pred_action": pm,
                    "exact_match": hit,
                }
            )

            marker = "✓" if pm == tm else "✗"
            tqdm.write(
                f"[{len(predictions) - 1}] {marker}  pred={pm!r} true={tm!r}\n"
                f"      PRED: {pred_text[:300]}\n"
                f"      TRUE: {true_text[:300]}"
            )
            bar.set_postfix(turns=len(true_actions), acc=f"{sum(exact_hits) / len(exact_hits):.3f}")

    n = len(true_actions)
    if n == 0:
        metrics = {
            "exact_match_acc": 0.0, "action_acc": 0.0,
            "action_f1": 0.0, "action_f1_weighted": 0.0, "num_turns": 0,
        }
    else:
        metrics = {
            "exact_match_acc": sum(exact_hits) / n,
            "action_acc": sum(p == t for p, t in zip(pred_actions, true_actions)) / n,
            "action_f1": sklearn_f1(true_actions, pred_actions, average="macro", zero_division=0),
            "action_f1_weighted": sklearn_f1(
                true_actions, pred_actions, average="weighted", zero_division=0
            ),
            "num_turns": n,
        }
    metrics["generation_time_sec"] = round(gen_time, 3)
    return metrics, predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference / eval for buyer simulation model.")
    parser.add_argument("--config", default="user_model/sft/config.yaml")
    parser.add_argument("--output_dir", default=None,
                        help="Override cfg.output_dir (checkpoint resolution and default pred_out).")
    parser.add_argument("--adapter_path", default=None,
                        help="Override checkpoint; else auto-resolve the best one.")
    parser.add_argument("--base_only", action="store_true",
                        help="Evaluate the original base model with no LoRA adapter "
                             "(pre-finetuning baseline).")
    parser.add_argument("--data", default=None,
                        help="Local .jsonl/.json path OR HF Hub dataset id. Else fall back to config.")
    parser.add_argument("--split", default="test", help="HF Hub split to load (default: test).")
    parser.add_argument(
        "--all_windows",
        action="store_true",
        help="Evaluate every overlapping #wN record. By default, windowed datasets are "
             "filtered to #w0; non-windowed datasets are unchanged.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of eval records (quick checks).")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Generation budget per turn. 256 truncated ~4%% of turns mid-JSON "
                             "(unparseable => scored as a miss); targets are short but a trailing "
                             "`url` field can be long. Costs nothing against max_length=32768.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decoding temperature; 0.0 = greedy (deterministic), > 0 = sampling.")
    parser.add_argument("--backend", choices=["vllm", "sglang", "hf"], default="vllm",
                        help="Generation backend: 'vllm' / 'sglang' (fast, batched) or "
                             "'hf' (transformers single-sample fallback).")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="Tensor-parallel size (number of GPUs to shard the model across). "
                             "Applies to --backend vllm/sglang only. To pick *which* physical "
                             "GPUs are used, set the CUDA_VISIBLE_DEVICES env var before running.")
    parser.add_argument("--pred_out", default=None,
                        help="Predictions output path (default: <output_dir>/eval_predictions.jsonl).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.base_only:
        if args.adapter_path:
            raise ValueError("--base_only and --adapter_path are mutually exclusive.")
        checkpoint = None
        logger.info("Evaluating the original base model (no LoRA adapter): %s",
                    cfg.model_name_or_path)
    else:
        checkpoint = resolve_checkpoint(cfg.output_dir, args.adapter_path)

    records = load_eval_data(cfg, args.data, args.split)
    if not args.all_windows:
        records = filter_w0_records(records)
    logger.info("Loaded %d eval records", len(records))

    if args.backend == "vllm":
        metrics, predictions = evaluate_vllm(
            cfg, records, checkpoint_path=checkpoint,
            max_new_tokens=args.max_new_tokens, limit=args.limit, temperature=args.temperature,
            tp_size=args.tp_size,
        )
    elif args.backend == "sglang":
        metrics, predictions = evaluate_sglang(
            cfg, records, checkpoint_path=checkpoint,
            max_new_tokens=args.max_new_tokens, limit=args.limit, temperature=args.temperature,
            tp_size=args.tp_size,
        )
    else:
        model, tokenizer = load_model(cfg, checkpoint)
        metrics, predictions = evaluate(
            model, tokenizer, records, max_new_tokens=args.max_new_tokens,
            limit=args.limit, temperature=args.temperature, max_length=cfg.max_length,
        )

    print("=== Eval metrics ===")
    print(json.dumps(metrics, indent=2))

    default_pred_name = "eval_predictions_base.jsonl" if args.base_only else "eval_predictions.jsonl"
    pred_out = args.pred_out or str(Path(cfg.output_dir) / default_pred_name)
    Path(pred_out).parent.mkdir(parents=True, exist_ok=True)
    with open(pred_out, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d predictions to %s", len(predictions), pred_out)
    # write the metrics to a JSON file for record
    metrics_out = Path(pred_out).with_suffix(".metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
