"""Recover the exact OPeRA sessions used by an existing SFT Hub dataset.

``user_model.sft.opera_data_prep`` publishes sliding windows whose ids are
``<OPeRA session_id>#w<N>``.  This script uses those ids as the selection list,
joins them back to ``NEU-HAI/OPeRA``, and publishes the selected sessions in the
normalized ``action``/``user`` layout consumed by ``user_model.rl.data_prep``.

The default command recreates the 50 training sessions in
``<huggingface_repo_id>/opera_50`` and publishes them as the ``train`` split of
``<huggingface_repo_id>/opera-50``::

    python -m user_model.rl.opera_data_prep \
      --output-repo <huggingface_repo_id>/opera-50 \
      --push-to-hub

The script verifies that rebuilding Method-1 windows from the selected raw
rows reproduces every reference row before it writes or publishes anything.
Use ``--no-verify-reference-messages`` only if the reference dataset was made
with a different prompt/template implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from user_model.sft.opera_data_prep import (
    _assistant_payload,
    _build_session,
    _intent_from_session,
    _persona_from_user,
)
from user_model.sft.data_loader import Session, session_to_records

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

WINDOW_SUFFIX = re.compile(r"#w\d+$")


def original_session_id(window_session_id: object) -> str:
    """Strip only the terminal window suffix added by ``session_to_records``."""
    value = str(window_session_id or "")
    return WINDOW_SUFFIX.sub("", value)


def reference_session_ids(reference_rows: Iterable[dict[str, Any]]) -> set[str]:
    """Return the unique original session ids represented by SFT windows."""
    session_ids: set[str] = set()
    for row in reference_rows:
        session_id = original_session_id(row.get("session_id"))
        if not session_id:
            raise ValueError("Reference dataset contains a row without session_id")
        session_ids.add(session_id)
    if not session_ids:
        raise ValueError("Reference dataset contains no sessions")
    return session_ids


def _normalized_action_row(
    row: dict[str, Any], payload: str, store_id: str
) -> dict[str, Any]:
    parsed = json.loads(payload)
    return {
        "store_id": store_id,
        "session_id": str(row["session_id"]),
        "timestamp": str(row.get("timestamp") or ""),
        "action_type": str(parsed.get("action") or ""),
        "target": str(parsed.get("target") or ""),
        "rationale": str(parsed.get("rationale") or ""),
        "input_text": str(parsed.get("text") or ""),
        "clicked_url": str(parsed.get("url") or ""),
        "simplified_dom": str(row.get("simplified_html") or ""),
        "action_json": payload,
    }


def build_matched_rows(
    reference_rows: Iterable[dict[str, Any]],
    action_rows: Iterable[dict[str, Any]],
    user_rows: Iterable[dict[str, Any]],
    session_rows: Iterable[dict[str, Any]],
    *,
    store_id: str = "opera",
    min_steps: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Session]]:
    """Join reference ids to raw OPeRA and emit normalized action/user rows."""
    selected = reference_session_ids(reference_rows)
    raw_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        session_id = str(row.get("session_id") or "")
        if session_id in selected:
            raw_actions[session_id].append(dict(row))

    session_users = {
        str(row.get("session_id") or ""): str(row.get("user_id") or "")
        for row in session_rows
        if str(row.get("session_id") or "") in selected
    }
    needed_user_ids = set(session_users.values())
    users_by_id = {
        str(row.get("user_id") or ""): dict(row)
        for row in user_rows
        if str(row.get("user_id") or "") in needed_user_ids
    }

    missing_actions = selected - raw_actions.keys()
    missing_sessions = selected - session_users.keys()
    missing_users = {
        sid for sid, user_id in session_users.items() if user_id not in users_by_id
    }
    if missing_actions or missing_sessions or missing_users:
        raise ValueError(
            "Could not join all reference sessions to OPeRA: "
            f"missing actions={sorted(missing_actions)}, "
            f"session metadata={sorted(missing_sessions)}, "
            f"users for sessions={sorted(missing_users)}"
        )

    normalized_actions: list[dict[str, Any]] = []
    normalized_users: list[dict[str, Any]] = []
    built_sessions: dict[str, Session] = {}
    for session_id in sorted(selected):
        rows = sorted(raw_actions[session_id], key=lambda row: str(row.get("timestamp") or ""))
        user_id = session_users[session_id]
        persona = _persona_from_user(users_by_id[user_id])
        session = _build_session(session_id, rows, persona)
        if session is None or len(session.steps) < min_steps:
            step_count = len(session.steps) if session is not None else 0
            raise ValueError(
                f"Reference session {session_id!r} rebuilt with {step_count} valid "
                f"step(s), fewer than min_steps={min_steps}"
            )
        built_sessions[session_id] = session
        normalized_users.append(
            {
                "store_id": store_id,
                "session_id": session_id,
                "user_id": user_id,
                "persona": persona,
                "intent": _intent_from_session(rows),
            }
        )
        for row in rows:
            payload = _assistant_payload(row)
            if payload is not None:
                normalized_actions.append(
                    _normalized_action_row(row, payload, store_id)
                )

    return normalized_actions, normalized_users, built_sessions


def verify_reference_windows(
    reference_rows: Iterable[dict[str, Any]],
    sessions: dict[str, Session],
    *,
    window_size: int = 15,
    stride: int = 5,
) -> None:
    """Require regenerated SFT windows to equal the existing reference rows."""
    expected = {
        row["session_id"]: row
        for session in sessions.values()
        for row in session_to_records(session, window_size, stride)
    }
    seen: set[str] = set()
    for reference in reference_rows:
        window_id = str(reference.get("session_id") or "")
        generated = expected.get(window_id)
        if generated is None:
            raise ValueError(f"Reference window {window_id!r} was not regenerated")
        if generated["messages"] != reference.get("messages"):
            raise ValueError(
                f"Regenerated messages differ from reference window {window_id!r}. "
                "Check window_size, stride, and prompt code versions."
            )
        seen.add(window_id)
    extra = expected.keys() - seen
    if extra:
        raise ValueError(f"Regeneration produced windows absent from reference: {sorted(extra)}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_opera_rl_source(
    *,
    reference_repo: str = "<huggingface_repo_id>/opera_50",
    reference_split: str = "train",
    source_repo: str = "NEU-HAI/OPeRA",
    source_config: str = "filtered",
    source_split: str = "train",
    output_dir: str | Path = "data/opera-50-source",
    output_repo: str | None = None,
    output_split: str = "train",
    store_id: str = "opera",
    expected_session_count: int | None = 50,
    min_steps: int = 2,
    window_size: int = 15,
    stride: int = 5,
    verify_reference_messages: bool = True,
    push_to_hub: bool = False,
    private: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Build, audit, optionally publish, and return an OPeRA match manifest."""
    from datasets import Dataset, DatasetDict, load_dataset

    logger.info("Loading reference %s split %s", reference_repo, reference_split)
    reference = list(load_dataset(reference_repo, split=reference_split, token=token))
    selected = reference_session_ids(reference)
    if expected_session_count is not None and len(selected) != expected_session_count:
        raise ValueError(
            f"Expected {expected_session_count} unique reference sessions, found {len(selected)}"
        )

    logger.info("Loading raw %s %s configurations", source_repo, source_config)
    actions = load_dataset(
        source_repo, f"{source_config}_action", split=source_split, token=token
    )
    users = load_dataset(
        source_repo, f"{source_config}_user", split=source_split, token=token
    )
    sessions = load_dataset(
        source_repo, f"{source_config}_session", split=source_split, token=token
    )
    action_rows, user_rows, built_sessions = build_matched_rows(
        reference,
        actions,
        users,
        sessions,
        store_id=store_id,
        min_steps=min_steps,
    )
    if verify_reference_messages:
        verify_reference_windows(
            reference, built_sessions, window_size=window_size, stride=stride
        )
        logger.info("Verified %d reference windows exactly", len(reference))

    output = Path(output_dir)
    action_path = output / f"action_{output_split}.jsonl"
    user_path = output / f"user_{output_split}.jsonl"
    _write_jsonl(action_path, action_rows)
    _write_jsonl(user_path, user_rows)
    manifest = {
        "reference_repo": reference_repo,
        "reference_split": reference_split,
        "source_repo": source_repo,
        "source_config": source_config,
        "source_split": source_split,
        "output_repo": output_repo,
        "output_split": output_split,
        "store_id": store_id,
        "session_count": len(built_sessions),
        "action_count": len(action_rows),
        "reference_window_count": len(reference),
        "reference_messages_verified": verify_reference_messages,
        "session_ids": sorted(built_sessions),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if push_to_hub:
        if not output_repo:
            raise ValueError("output_repo is required when push_to_hub=True")
        for config_name, rows in (("action", action_rows), ("user", user_rows)):
            logger.info("Pushing %s/%s to %s", config_name, output_split, output_repo)
            DatasetDict({output_split: Dataset.from_list(rows)}).push_to_hub(
                output_repo,
                config_name=config_name,
                private=private,
                token=token,
                commit_message=(
                    f"Recover {len(built_sessions)} exact OPeRA sessions for {config_name}"
                ),
            )
    logger.info(
        "Prepared %d exact OPeRA sessions / %d actions in %s",
        len(built_sessions),
        len(action_rows),
        output,
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover exact OPeRA sessions from an existing windowed SFT dataset."
    )
    parser.add_argument("--reference-repo", "--reference_repo", default="<huggingface_repo_id>/opera_50")
    parser.add_argument("--reference-split", "--reference_split", default="train")
    parser.add_argument("--source-repo", "--source_repo", default="NEU-HAI/OPeRA")
    parser.add_argument("--source-config", "--source_config", choices=["filtered", "full"], default="filtered")
    parser.add_argument("--source-split", "--source_split", default="train")
    parser.add_argument("--output-dir", "--output_dir", default="data/opera-50-source")
    parser.add_argument("--output-repo", "--output_repo", default="<huggingface_repo_id>/opera-50")
    parser.add_argument("--output-split", "--output_split", default="train")
    parser.add_argument("--store-id", "--store_id", default="opera")
    parser.add_argument("--expected-session-count", "--expected_session_count", type=int, default=50)
    parser.add_argument("--min-steps", "--min_steps", type=int, default=2)
    parser.add_argument("--window-size", "--window_size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--no-verify-reference-messages",
        dest="verify_reference_messages",
        action="store_false",
    )
    parser.add_argument("--push-to-hub", "--push_to_hub", action="store_true")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True, verify_reference_messages=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = prepare_opera_rl_source(**vars(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
