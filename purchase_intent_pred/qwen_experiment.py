"""Run the final Qwen3.5 purchase-intent experiment.

Qwen3.5-4B is QLoRA-fine-tuned under TRTR, TSTR, and TRSTR and evaluated on
the same imbalanced real test set. Input is the raw chronological action
sequence. Inference is greedy free generation of exactly one JSON action; no
candidate scoring, probability threshold, or calibration set is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


logger = logging.getLogger("pip.qwen_experiment")

MODEL_ID = "Qwen/Qwen3.5-4B"
PURCHASE = "purchase"
TERMINATE = "terminate"
VALID_OUTCOMES = (PURCHASE, TERMINATE)
CONDITIONS = ("trtr", "tstr", "trstr")
DEFAULT_SEEDS = (13, 17, 23, 29, 31)
EVALUATION_METHOD = "free_generation"
EVENT_FIELDS = ("action",)
ACTION_REPRESENTATION = "raw"
AGGREGATE_METRICS = ("f1",)

SYSTEM_PROMPT = (
    "You predict the final action of an online shopping session. "
    "Use only the observed history. Return exactly one JSON object and no "
    "explanation: {\"action\": \"purchase\"} or "
    "{\"action\": \"terminate\"}."
)


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("outcome") not in VALID_OUTCOMES:
                raise ValueError(f"Invalid outcome at {path}:{line_number}")
            rows.append(row)
    return rows


def _protocol_record(record: dict) -> dict:
    """Return only fields that can affect final action-only training/inference."""
    return {
        "session_id": str(record.get("session_id") or ""),
        "source": str(record.get("source") or ""),
        "outcome": record.get("outcome"),
        "actions": [
            str(event.get("action") or "")
            for event in record.get("events", [])
            if event.get("action")
        ],
    }


def records_fingerprint(records: Iterable[dict]) -> str:
    payload = json.dumps(
        [_protocol_record(record) for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def session_user_text(record: dict) -> str:
    history = [
        {"action": event["action"]}
        for event in record.get("events", [])
        if event.get("action")
    ]
    return (
        "Observed session history (chronological JSON):\n"
        + json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        + "\nPredict the unobserved final action."
    )


def chat_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": session_user_text(record)}],
        },
    ]


def json_completion(outcome: str) -> str:
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"Unknown outcome: {outcome}")
    return json.dumps({"action": outcome}, separators=(",", ":"))


def _render_prompt(processor, record: dict) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return processor.apply_chat_template(chat_messages(record), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        return processor.apply_chat_template(chat_messages(record), **kwargs)


class OutcomeTokenDataset:
    """Causal-LM data whose loss applies only to the final JSON response."""

    def __init__(self, records: list[dict], processor, max_length: int):
        self.items: list[dict[str, list[int]]] = []
        tokenizer = processor.tokenizer
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Qwen tokenizer has no EOS token")
        for record in records:
            prompt_ids = tokenizer(
                _render_prompt(processor, record), add_special_tokens=False
            )["input_ids"]
            answer_ids = tokenizer(
                json_completion(record["outcome"]), add_special_tokens=False
            )["input_ids"] + [eos_id]
            if len(answer_ids) >= max_length:
                raise ValueError("--max_length is too small for the JSON response")
            prompt_ids = prompt_ids[-(max_length - len(answer_ids)) :]
            self.items.append(
                {
                    "input_ids": prompt_ids + answer_ids,
                    "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
                    "labels": [-100] * len(prompt_ids) + answer_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CausalPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _device_for_inputs(model):
    import torch

    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except (AttributeError, TypeError):
        pass
    for parameter in model.parameters():
        if parameter.device != torch.device("meta"):
            return parameter.device
    raise RuntimeError("Could not locate a non-meta model device")


def load_model_and_processor(
    adapter_path: str | Path | None = None,
    trainable_adapter: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load the fixed 4-bit Qwen3.5-4B model and optional LoRA adapter."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("The finalized 4-bit Qwen experiment requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=dtype,
        quantization_config=quantization_config,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=trainable_adapter
        )
    elif trainable_adapter:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        model.config.text_config.use_cache = False
    return model, processor


def training_manifest(records: list[dict], condition: str, seed: int, args) -> dict:
    """Fingerprint all settings that make a saved adapter reusable."""
    return {
        "model_id": MODEL_ID,
        "protocol": "action_only_free_generation",
        "condition": condition,
        "seed": seed,
        "training_records_fingerprint": records_fingerprint(records),
        "n_train": len(records),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "gradient_checkpointing": args.gradient_checkpointing,
    }


def train_adapter(
    records: list[dict], condition: str, seed: int, adapter_dir: Path, args
) -> None:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model, processor = load_model_and_processor(
        trainable_adapter=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    dataset = OutcomeTokenDataset(records, processor, args.max_length)
    collator = CausalPaddingCollator(processor.tokenizer.pad_token_id)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    bf16 = torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_dir / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=bf16,
            fp16=not bf16,
            optim="paged_adamw_8bit",
            seed=seed,
            data_seed=seed,
        ),
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    (adapter_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest(records, condition, seed, args), indent=2),
        encoding="utf-8",
    )
    del trainer, dataset, model, processor
    gc.collect()
    torch.cuda.empty_cache()


def parse_generated_action(text: str) -> str | None:
    """Accept only the exact one-field JSON schema used by the experiment."""
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"action"}:
        return None
    action = value.get("action")
    return action if action in VALID_OUTCOMES else None


def generate_records(
    model,
    processor,
    records: list[dict],
    max_length: int,
    max_new_tokens: int,
) -> list[dict]:
    """Greedily generate one final JSON action for every test session."""
    import torch

    if max_new_tokens <= 0 or max_new_tokens >= max_length:
        raise ValueError("--generation_max_new_tokens must be in (0, max_length)")
    tokenizer = processor.tokenizer
    device = _device_for_inputs(model)
    model.eval()
    predictions: list[dict] = []
    for index, record in enumerate(records):
        prompt_ids = tokenizer(
            _render_prompt(processor, record), add_special_tokens=False
        )["input_ids"]
        prompt_ids = prompt_ids[-max(1, max_length - max_new_tokens) :]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        action = parse_generated_action(generated_text)
        predictions.append(
            {
                "session_id": record["session_id"],
                "label": int(record["outcome"] == PURCHASE),
                "gold": {"action": record["outcome"]},
                "prediction": {"action": action} if action is not None else None,
                "generated_text": generated_text,
                "valid_generation": action is not None,
            }
        )
        if (index + 1) % 25 == 0:
            logger.info("Generated %d/%d test sessions", index + 1, len(records))
    torch.cuda.empty_cache()
    return predictions


def generation_metrics(predictions: list[dict]) -> dict[str, object]:
    """Score hard actions; invalid JSON is always an incorrect prediction."""
    tp = fp = fn = tn = invalid = predicted_purchase = 0
    for row in predictions:
        label = int(row["label"])
        prediction = row.get("prediction")
        action = prediction.get("action") if isinstance(prediction, dict) else None
        if action not in VALID_OUTCOMES:
            invalid += 1
            if label:
                fn += 1
            else:
                fp += 1
            continue
        predicted_label = int(action == PURCHASE)
        predicted_purchase += predicted_label
        if label and predicted_label:
            tp += 1
        elif label:
            fn += 1
        elif predicted_label:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / len(predictions),
        "predicted_purchase": predicted_purchase,
        "predicted_purchase_rate": predicted_purchase / len(predictions),
        "invalid_generation_rate": invalid / len(predictions),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "invalid": invalid,
        },
    }


def bootstrap_mean_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int | str | None]:
    """Summarize independent training-seed results with a percentile CI."""
    values = list(values)
    if not values:
        return {
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "confidence": confidence,
            "method": "percentile_bootstrap_over_training_seeds",
            "n_seeds": 0,
        }
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": None,
            "ci_low": values[0],
            "ci_high": values[0],
            "confidence": confidence,
            "method": "percentile_bootstrap_over_training_seeds",
            "n_seeds": 1,
        }
    rng = random.Random(seed)
    bootstrap = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(n_bootstrap)
    )
    alpha = (1 - confidence) / 2
    low_index = max(0, int(alpha * n_bootstrap))
    high_index = min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap) - 1)
    return {
        "mean": mean(values),
        "std": stdev(values),
        "ci_low": bootstrap[low_index],
        "ci_high": bootstrap[high_index],
        "confidence": confidence,
        "method": "percentile_bootstrap_over_training_seeds",
        "n_seeds": len(values),
    }


def aggregate_seed_metrics(per_seed: list[dict]) -> dict:
    return {
        metric: bootstrap_mean_ci([row["metrics"][metric] for row in per_seed])
        for metric in AGGREGATE_METRICS
    }


def aggregate_paired_difference(baseline: list[dict], candidate: list[dict]) -> dict:
    baseline_by_seed = {row["seed"]: row["metrics"] for row in baseline}
    candidate_by_seed = {row["seed"]: row["metrics"] for row in candidate}
    if baseline_by_seed.keys() != candidate_by_seed.keys():
        raise ValueError("Condition seeds differ; paired comparison is invalid")
    return {
        metric: bootstrap_mean_ci(
            [
                candidate_by_seed[seed][metric] - baseline_by_seed[seed][metric]
                for seed in sorted(baseline_by_seed)
            ]
        )
        for metric in AGGREGATE_METRICS
    }


def validate_experiment_data(
    trtr_records: list[dict] | None,
    tstr_records: list[dict] | None,
    trstr_records: list[dict] | None,
    test_records: list[dict],
) -> None:
    def session_ids(records: list[dict], split: str, duplicates: bool = False) -> list[str]:
        ids = [str(row.get("session_id") or "") for row in records]
        if not all(ids):
            raise ValueError(f"{split} contains a missing session_id")
        if not duplicates and len(ids) != len(set(ids)):
            raise ValueError(f"{split} contains duplicate session IDs")
        return ids

    test_ids = set(session_ids(test_records, "test"))
    n_purchase = sum(row["outcome"] == PURCHASE for row in test_records)
    n_terminate = len(test_records) - n_purchase
    expected_purchase = len(test_records) // 11
    if n_purchase != expected_purchase or n_terminate != len(test_records) - expected_purchase:
        raise ValueError("The final real test set must use the fixed 10:1 imbalance")

    training_ids: list[set[str]] = []
    if trtr_records is not None:
        training_ids.append(set(session_ids(trtr_records, "trtr_train")))
    if tstr_records is not None:
        training_ids.append(set(session_ids(tstr_records, "tstr_train")))
    if len(training_ids) == 2 and training_ids[0] != training_ids[1]:
        raise ValueError("TRTR and TSTR must contain identical training IDs")
    if trstr_records is not None:
        ids = session_ids(trstr_records, "trstr_train", duplicates=True)
        counts = Counter(ids)
        sources: dict[str, set[str]] = {}
        for row, session_id in zip(trstr_records, ids):
            sources.setdefault(session_id, set()).add(str(row.get("source") or ""))
        if set(counts.values()) != {2}:
            raise ValueError("TRSTR must contain exactly two examples per session ID")
        if any(source_set != {"real", "synthetic"} for source_set in sources.values()):
            raise ValueError("Each TRSTR ID must contain real and synthetic examples")
        trstr_ids = set(counts)
        if training_ids and trstr_ids != training_ids[0]:
            raise ValueError("TRSTR IDs must match the TRTR/TSTR paired IDs")
        training_ids.append(trstr_ids)
    if any(ids & test_ids for ids in training_ids):
        raise ValueError("A training session ID leaked into the test set")


def _evaluation_config(args) -> dict:
    return {
        "model_id": MODEL_ID,
        "evaluation_method": EVALUATION_METHOD,
        "event_fields": list(EVENT_FIELDS),
        "action_representation": ACTION_REPRESENTATION,
        "max_length": args.max_length,
        "generation_max_new_tokens": args.generation_max_new_tokens,
    }


def _cached_result_is_compatible(
    result: dict,
    condition: str,
    seed: int,
    test_fingerprint: str,
    evaluation_config: dict,
) -> bool:
    return (
        result.get("condition") == condition
        and result.get("seed") == seed
        and result.get("test_fingerprint") == test_fingerprint
        and result.get("evaluation_config") == evaluation_config
        and isinstance(result.get("metrics"), dict)
        and "f1" in result["metrics"]
    )


def evaluate_adapter(
    test_records: list[dict], adapter_dir: Path, condition: str, seed: int, args
) -> dict:
    import torch

    model, processor = load_model_and_processor(adapter_path=adapter_dir)
    predictions = generate_records(
        model,
        processor,
        test_records,
        max_length=args.max_length,
        max_new_tokens=args.generation_max_new_tokens,
    )
    result = {
        "condition": condition,
        "seed": seed,
        "adapter": str(adapter_dir),
        "evaluation_method": EVALUATION_METHOD,
        "test_fingerprint": records_fingerprint(test_records),
        "evaluation_config": _evaluation_config(args),
        "n_test": len(test_records),
        "metrics": generation_metrics(predictions),
    }
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    (adapter_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def run(args) -> dict:
    if len(args.seeds) < 5 and not args.allow_fewer_seeds:
        raise ValueError(
            "The final experiment requires at least five seeds; use "
            "--allow_fewer_seeds only for smoke tests"
        )
    data_dir = Path(args.data_dir)
    test_records = load_jsonl(data_dir / "test.jsonl")
    train_by_condition = {
        condition: load_jsonl(data_dir / f"{condition}_train.jsonl")
        for condition in args.conditions
    }
    validate_experiment_data(
        train_by_condition.get("trtr"),
        train_by_condition.get("tstr"),
        train_by_condition.get("trstr"),
        test_records,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_fingerprint = records_fingerprint(test_records)
    evaluation_config = _evaluation_config(args)
    all_results: dict[str, dict] = {}

    for condition in args.conditions:
        train_records = train_by_condition[condition]
        per_seed: list[dict] = []
        for seed in args.seeds:
            adapter_dir = output_dir / condition / f"seed_{seed}"
            adapter_complete = (adapter_dir / "adapter_config.json").is_file()
            expected_manifest = training_manifest(train_records, condition, seed, args)
            saved_manifest = _read_json(adapter_dir / "training_manifest.json")
            if adapter_complete and saved_manifest != expected_manifest and not args.overwrite_existing:
                raise ValueError(
                    f"Existing adapter is stale or unverifiable: {adapter_dir}. "
                    "Use a new output directory or --overwrite_existing."
                )

            cached_result = _read_json(adapter_dir / "metrics.json")
            if (
                adapter_complete
                and not args.overwrite_existing
                and cached_result is not None
                and _cached_result_is_compatible(
                    cached_result,
                    condition,
                    seed,
                    test_fingerprint,
                    evaluation_config,
                )
            ):
                logger.info("Reusing completed seed: %s", adapter_dir)
                per_seed.append(cached_result)
                continue
            if args.eval_only and not adapter_complete:
                raise FileNotFoundError(f"Missing adapter for --eval_only: {adapter_dir}")
            if not args.eval_only and (not adapter_complete or args.overwrite_existing):
                train_adapter(train_records, condition, seed, adapter_dir, args)
            elif not args.eval_only:
                logger.info("Reusing compatible adapter: %s", adapter_dir)

            result = evaluate_adapter(test_records, adapter_dir, condition, seed, args)
            per_seed.append(result)
            logger.info("%s seed=%d metrics=%s", condition, seed, result["metrics"])
        all_results[condition] = {
            "n_train": len(train_records),
            "per_seed": per_seed,
            "aggregate": aggregate_seed_metrics(per_seed),
        }

    comparisons = (
        ("trtr", "tstr", "tstr_minus_trtr"),
        ("trtr", "trstr", "trstr_minus_trtr"),
        ("tstr", "trstr", "trstr_minus_tstr"),
    )
    for baseline, candidate, name in comparisons:
        if baseline in all_results and candidate in all_results:
            all_results[name] = aggregate_paired_difference(
                all_results[baseline]["per_seed"], all_results[candidate]["per_seed"]
            )

    final_f1_summary = {
        condition: {
            "mean": all_results[condition]["aggregate"]["f1"]["mean"],
            "std": all_results[condition]["aggregate"]["f1"]["std"],
            "n_seeds": all_results[condition]["aggregate"]["f1"]["n_seeds"],
        }
        for condition in CONDITIONS
        if condition in all_results
    }
    report = {
        "model_id": MODEL_ID,
        "protocol": "action_only_free_generation",
        "evaluation_method": EVALUATION_METHOD,
        "event_fields": list(EVENT_FIELDS),
        "action_representation": ACTION_REPRESENTATION,
        "seeds": args.seeds,
        "n_test": len(test_records),
        "test_purchase": sum(row["outcome"] == PURCHASE for row in test_records),
        "test_terminate": sum(row["outcome"] == TERMINATE for row in test_records),
        "test_purchase_rate": sum(
            row["outcome"] == PURCHASE for row in test_records
        )
        / len(test_records),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_checkpointing": args.gradient_checkpointing,
            "quantization": "4bit_nf4_double_quantization",
        },
        "confidence_interval": "95% percentile bootstrap over training seeds",
        "final_f1_summary": {
            "format": "mean and sample standard deviation across training seeds",
            "conditions": final_f1_summary,
        },
        "conditions": all_results,
    }
    (output_dir / "experiment_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Qwen3.5-4B TRTR/TSTR/TRSTR experiment"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--output_dir", default="purchase_intent_pred/qwen_runs_final"
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--allow_fewer_seeds", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--generation_max_new_tokens", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
