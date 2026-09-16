"""Build local next-action GRPO records from normalized Hub data.

Each assistant turn becomes one GRPO example.  The prompt is exactly the message
prefix used by ``user_model.inference`` (system/persona/intent, prior observed
pages and actions, and the current simplified HTML).  The held-out assistant
turn supplies reward labels only; it is never included in the prompt.

The current simplified HTML is preserved verbatim in the ``ui`` field for the
UI-grounding judge.

Primary usage:

  python -m user_model.rl.data_prep \
    --input-repo <huggingface_repo_id>/buyer-sim-50 \
    --input-split train+test \
    --output_dir data/rl/buyer-sim-50

The input repository must use the normalized format produced by
``user_model.publish_data_to_hf``: one row per action in the ``action`` configuration
and one row per session in ``user``. Every valid session in the selected input
split is flattened to turns and written to ``<output_dir>/train.jsonl``.
Nothing is uploaded; source matching or sampling happens upstream.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from user_model.sft.data_prep import reconstruct_sessions
from user_model.prompt_templates import normalize_assistant_payload, render_system_prompt
from user_model.sft.data_loader import Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def extract_ui(user_content: Any) -> str:
    """Return the simplified HTML exactly, without the prompt context marker."""
    if not isinstance(user_content, str):
        return ""
    prefix = "# context\n"
    return user_content[len(prefix):] if user_content.startswith(prefix) else user_content


def conversation_to_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one ``{session_id, messages}`` conversation into assistant turns."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    session_id = str(record.get("session_id", ""))
    turns: list[dict[str, Any]] = []
    assistant_idx = 0
    for idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        payload = normalize_assistant_payload(message.get("content", ""))
        if not payload or not isinstance(payload.get("action"), str):
            logger.warning("Skipping malformed assistant turn %s#t%d", session_id, assistant_idx)
            assistant_idx += 1
            continue
        current_user = next(
            (
                prior.get("content", "")
                for prior in reversed(messages[:idx])
                if isinstance(prior, dict) and prior.get("role") == "user"
            ),
            "",
        )
        turn = {
            "prompt": messages[:idx],
            "reference_rationale": str(payload.get("rationale", "")),
            "gt_action": str(payload["action"]).strip().lower(),
            "gt_target": str(payload.get("target", "") or ""),
            # Retain action-specific fields such as text/value/key/amount so the
            # W&B completion table can show the complete reference response.
            "ground_truth_completion": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ),
            "ui": extract_ui(current_user),
            "session_id": session_id,
            "turn_idx": assistant_idx,
        }
        if record.get("store_id") is not None:
            turn["store_id"] = str(record["store_id"])
        turns.append(turn)
        assistant_idx += 1
    return turns


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def records_to_turns(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for record in records:
        # Accept an already materialized turn dataset too, so the trainer can
        # read prepare_turn_data's local output without flattening it again.
        if all(
            key in record
            for key in ("prompt", "reference_rationale", "gt_action", "gt_target", "ui")
        ):
            turns.append(dict(record))
        else:
            turns.extend(conversation_to_turns(record))
    return turns


def load_turn_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = records_to_turns(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No valid assistant turns found in {path}.")
    logger.info("Built %d turn-level GRPO rows from %s", len(rows), path)
    return rows


def build_turn_datasets(
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    dataset_hub_id: Optional[str] = None,
):
    """Return turn-level ``(train, validation)`` Hugging Face datasets."""
    from datasets import Dataset, load_dataset

    if dataset_hub_id:
        source = load_dataset(dataset_hub_id)
        if "train" not in source:
            raise ValueError(f"Hub dataset {dataset_hub_id!r} has no train split")
        train_records = source["train"]
        val_records = source.get("validation")
        train_rows = records_to_turns(train_records)
        val_rows = records_to_turns(val_records) if val_records is not None else []
    else:
        if not train_data_path:
            raise ValueError("train_data_path is required when dataset_hub_id is not set")
        train_rows = load_turn_rows(train_data_path)
        val_rows = (
            load_turn_rows(val_data_path)
            if val_data_path and Path(val_data_path).exists()
            else []
        )
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows) if val_rows else None


def session_to_conversation(session: Session, store_id: str) -> dict[str, Any]:
    """Reconstruct the full Method-1 conversation for one normalized session."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": render_system_prompt(session.persona, session.intent),
        }
    ]
    for step in session.steps:
        messages.append({"role": "user", "content": f"# context\n{step.html}"})
        messages.append({"role": "assistant", "content": step.assistant_payload})
    return {
        "store_id": str(store_id),
        "session_id": session.session_id,
        "messages": messages,
    }


def prepare_turn_data(
    input_repo: str | None = None,
    output_dir: str | Path | None = None,
    input_split: str = "train+test",
    min_steps: int = 2,
    *,
    synthetic_repo: str | None = None,
    synthetic_split: str | None = None,
) -> Path:
    """Convert every valid session in a normalized input repo to RL turn JSONL.

    ``synthetic_repo`` and ``synthetic_split`` remain as keyword-only aliases
    for older callers. New code should use the source-neutral input names.
    """
    from datasets import load_dataset

    if input_repo is not None and synthetic_repo is not None and input_repo != synthetic_repo:
        raise ValueError("input_repo and legacy synthetic_repo refer to different repositories")
    if synthetic_repo is not None:
        input_repo = synthetic_repo
    if synthetic_split is not None:
        if input_split != "train+test" and input_split != synthetic_split:
            raise ValueError("input_split and legacy synthetic_split refer to different splits")
        input_split = synthetic_split
    if not input_repo:
        raise ValueError("input_repo is required")
    if output_dir is None:
        raise ValueError("output_dir is required")

    logger.info(
        "Loading normalized action/%s from %s",
        input_split,
        input_repo,
    )
    action_rows = load_dataset(input_repo, "action", split=input_split)
    logger.info(
        "Loading normalized user/%s from %s",
        input_split,
        input_repo,
    )
    user_rows = load_dataset(input_repo, "user", split=input_split)
    sessions_by_store = reconstruct_sessions(
        action_rows,
        user_rows,
        min_steps=min_steps,
    )
    conversations = [
        session_to_conversation(session, store_id)
        for store_id in sorted(sessions_by_store)
        for session in sessions_by_store[store_id]
    ]
    turns = records_to_turns(conversations)
    if not turns:
        raise ValueError("No turn rows were produced from the input repository")

    output = Path(output_dir)
    output_path = output / "train.jsonl"
    _write_jsonl(output_path, turns)
    store_counts = {
        store_id: len(sessions) for store_id, sessions in sessions_by_store.items()
    }
    manifest = {
        "input_repo": input_repo,
        "input_split": input_split,
        "min_steps": min_steps,
        "sessions_by_store": store_counts,
        "session_count": sum(store_counts.values()),
        "turn_count": len(turns),
        "output_file": output_path.name,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Prepared %d sessions / %d turns in %s",
        sum(store_counts.values()),
        len(turns),
        output,
    )
    return output_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a normalized Hub repo to local turn-level GRPO JSONL."
        )
    )
    parser.add_argument(
        "--input-repo",
        "--input_repo",
        "--synthetic-repo",
        "--synthetic_repo",
        dest="input_repo",
        required=True,
        help="Normalized Hub dataset containing action and user configurations.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument(
        "--input-split",
        "--input_split",
        "--synthetic-split",
        "--synthetic_split",
        dest="input_split",
        default="train+test",
    )
    parser.add_argument("--min-steps", "--min_steps", dest="min_steps", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = prepare_turn_data(
        input_repo=args.input_repo,
        output_dir=args.output_dir,
        input_split=args.input_split,
        min_steps=args.min_steps,
    )
    print(path)
