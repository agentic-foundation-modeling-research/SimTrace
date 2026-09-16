"""Convert a sampled normalized buyer-sim Hub dataset to local SFT JSONL.

The normalized publication format stores one row per action in the ``action``
configuration and one row per session in ``user``.  The existing trainer needs
Method-1 rows with ``{"session_id", "messages"}``. Session selection belongs to
``user_model.sft.data_sampling``; this script converts every valid session in the
sampled repository and writes ``<output_dir>/sft_train.jsonl``. It neither
resamples nor mixes in another dataset.

Example::

    python -m user_model.sft.data_prep \
        --synthetic_repo <huggingface_repo_id>/buyer-sim-50 \
        --output_dir data/sft/buyer-sim-50
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from user_model.prompt_templates import normalize_assistant_payload
from user_model.sft.data_loader import Session, Step, session_to_records

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _string(value: object) -> str:
    return "" if value is None else str(value)


def _session_key(row: dict) -> tuple[str, str]:
    return _string(row.get("store_id")), _string(row.get("session_id"))


def reconstruct_sessions(
    action_rows: Iterable[dict],
    user_rows: Iterable[dict],
    min_steps: int = 2,
) -> dict[str, list[Session]]:
    """Reconstruct sessions grouped by store id from normalized Hub rows."""
    users: dict[tuple[str, str], dict] = {}
    for row in user_rows:
        key = _session_key(row)
        if not key[1]:
            logger.warning("Skipping user row without session_id: %s", row)
            continue
        users[key] = row

    actions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in action_rows:
        key = _session_key(row)
        if not key[1]:
            logger.warning("Skipping action row without session_id")
            continue
        actions[key].append(row)

    by_store: dict[str, list[Session]] = defaultdict(list)
    for key in sorted(actions):
        store_id, session_id = key
        user = users.get(key)
        if user is None:
            logger.warning(
                "Skipping store_id=%s session_id=%s: no matching user row",
                store_id,
                session_id,
            )
            continue
        steps: list[Step] = []
        for row in sorted(actions[key], key=lambda item: _string(item.get("timestamp"))):
            raw_payload = row.get("action_json")
            payload = normalize_assistant_payload(raw_payload)
            if not payload:
                logger.warning(
                    "Skipping invalid action_json in store_id=%s session_id=%s",
                    store_id,
                    session_id,
                )
                continue
            rationale = _string(payload.get("rationale"))
            action = {field: value for field, value in payload.items() if field != "rationale"}
            steps.append(
                Step(
                    rationale=rationale,
                    action=action,
                    assistant_payload=json.dumps(payload, ensure_ascii=False),
                    html=_string(row.get("simplified_dom")),
                )
            )
        if len(steps) < min_steps:
            logger.info(
                "Dropping store_id=%s session_id=%s: %d valid steps < min_steps=%d",
                store_id,
                session_id,
                len(steps),
                min_steps,
            )
            continue
        by_store[store_id].append(
            Session(
                session_id=session_id,
                persona=_string(user.get("persona")),
                intent=_string(user.get("intent")),
                steps=steps,
            )
        )
    return dict(by_store)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _method1_rows(
    sessions_by_store: dict[str, list[Session]], window_size: int, stride: int
) -> list[dict]:
    return [
        {**row, "store_id": store_id}
        for store_id in sorted(sessions_by_store)
        for session in sessions_by_store[store_id]
        for row in session_to_records(session, window_size, stride)
    ]


def prepare_sft_data(
    synthetic_repo: str,
    output_dir: str | Path,
    synthetic_split: str = "train+test",
    min_steps: int = 2,
    window_size: int = 15,
    stride: int = 5,
) -> Path:
    """Download a sampled normalized repo and convert all valid sessions to SFT."""
    from datasets import load_dataset

    logger.info("Loading normalized synthetic action/%s from %s", synthetic_split, synthetic_repo)
    action_ds = load_dataset(synthetic_repo, "action", split=synthetic_split)
    logger.info("Loading normalized synthetic user/%s from %s", synthetic_split, synthetic_repo)
    user_ds = load_dataset(synthetic_repo, "user", split=synthetic_split)
    sessions_by_store = reconstruct_sessions(action_ds, user_ds, min_steps=min_steps)
    synthetic_rows = _method1_rows(sessions_by_store, window_size, stride)
    if not synthetic_rows:
        raise ValueError("No synthetic Method-1 rows were produced")

    output = Path(output_dir)
    synthetic_path = output / "sft_train.jsonl"
    synthetic_count = _write_jsonl(synthetic_path, synthetic_rows)
    store_counts = {
        store_id: len(sessions) for store_id, sessions in sessions_by_store.items()
    }
    manifest = {
        "synthetic_repo": synthetic_repo,
        "synthetic_split": synthetic_split,
        "min_steps": min_steps,
        "window_size": window_size,
        "stride": stride,
        "sessions_by_store": store_counts,
        "session_count": sum(store_counts.values()),
        "row_count": synthetic_count,
        "output_file": synthetic_path.name,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Prepared %d sampled synthetic sessions as %d SFT rows in %s",
        sum(store_counts.values()),
        synthetic_count,
        output,
    )
    return synthetic_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a sampled normalized buyer-sim Hub repo to local SFT JSONL."
    )
    parser.add_argument("--synthetic_repo", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--synthetic_split", default="train+test")
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = prepare_sft_data(
        synthetic_repo=args.synthetic_repo,
        output_dir=args.output_dir,
        synthetic_split=args.synthetic_split,
        min_steps=args.min_steps,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(path)
