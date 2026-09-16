"""Adapt OPeRA (NEU-HAI/OPeRA) to the multi-turn SFT format.

OPeRA = Observation, Persona, Rationale, Action — real human Amazon shopping
sessions. It already carries exactly the four signals the user model trains on,
so there is nothing to *generate*: this script replaces the browser-simulation
data-generation stage and emits the same `{"session_id", "messages": [...]}`
JSONL that `data_prep.py` produces, which `trainer.py` consumes unchanged.

Mapping
-------
  system    <- render_system_prompt(persona, intent)
              persona  : user-table survey + processed interview transcript
              intent   : synthesized from the session's products / search terms
                         (OPeRA has no explicit per-session intent — see report)
  user      <- "# context\n<simplified_html>"   (one per action, preserved in full)
  assistant <- flat JSON {"rationale", "action": "<type>", "target", "description", ...}

Each session is split into overlapping fixed-step sliding windows (like
`data_prep.py`) so long sessions contribute every step instead of being
truncated at a step cap.

Usage
-----
  python -m user_model.sft.opera_data_prep \
      --config filtered --output_dir data/opera \
      [--window_size 15] [--stride 5] \
      [--val_ratio 0.1] [--sample_size 50] \
      [--push_to_hub --repo_id your-org/opera_sim [--private]]

``--sample_size`` caps how many *training* sessions the dataset keeps (low-real-data
experiments). Sampling happens after the validation split, so validation and test are
identical across every sample size and runs stay comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from user_model.sft.data_loader import Session, Step, session_to_records

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _safe_json(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# Action mapping: OPeRA row -> flat action dict (repo schema, "action" = type)
# Rules cover BOTH the filtered (click/input/terminate) and full
# (+ scroll/navigation/tab_activate) action spaces so switching --config
# doesn't silently drop steps.
# --------------------------------------------------------------------------- #
def _flat_action(row: dict) -> dict | None:
    action_type = (row.get("action_type") or "").strip()
    semantic_id = row.get("semantic_id") or ""

    if action_type == "click":
        return {"action": "click", "target": semantic_id,
                "description": f"Clicking the {row.get('click_type') or 'element'} element."}
    if action_type == "input":
        return {"action": "type", "target": semantic_id,
                "text": row.get("input_text") or "",
                "description": "Typing into the input field."}
    if action_type == "terminate":
        return {"action": "terminate", "description": "Ending the session."}
    if action_type == "scroll":
        detail = _safe_json(row.get("scroll_detail"), {})
        dy = str(detail.get("scroll_distance_y", "") if isinstance(detail, dict) else "")
        direction = "up" if dy.startswith("-") else "down"
        return {"action": "scroll", "target": semantic_id, "direction": direction,
                "description": f"Scrolling {direction}."}
    if action_type == "navigation":
        nav = (row.get("navigation_type") or "").strip().lower()
        method = {"back": "back", "forward": "forward", "reload": "refresh"}.get(nav, "goto_url")
        act = {"action": method, "description": f"Navigating ({nav or 'new page'})."}
        if method == "goto_url":
            act["url"] = row.get("url") or ""
        return act
    if action_type == "tab_activate":
        return {"action": "switch_tab", "description": "Switching browser tab."}
    return None


def _assistant_payload(row: dict) -> str | None:
    action = _flat_action(row)
    if action is None:
        logger.debug("Unmapped action_type: %r", row.get("action_type"))
        return None
    payload = {"rationale": (row.get("rationale") or "").strip(), **action}
    return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Persona / intent
# --------------------------------------------------------------------------- #
def _persona_from_user(user_row: dict | None) -> str:
    if not user_row:
        return "A general online shopper. No detailed persona available."
    processed = (user_row.get("interview_transcript_processed") or "").strip()
    if processed:
        return processed
    survey = _safe_json(user_row.get("survey"), {})
    demo = survey.get("Demographic Information", {})
    pref = survey.get("Shopping Preference", {})
    desc = survey.get("Self Description", {}).get("Two sentence description", "")
    parts = [desc] if desc else []
    if demo:
        parts.append("Demographics: " + json.dumps(demo, ensure_ascii=False))
    if pref:
        parts.append("Shopping preferences: " + json.dumps(pref, ensure_ascii=False))
    return "\n".join(parts) or "A general online shopper."


def _intent_from_session(rows: list[dict]) -> str:
    """Synthesize a session intent from product titles / search terms.

    OPeRA does not record a per-session intent, so we approximate it from the
    products touched and search terms typed during the session.
    """
    titles: list[str] = []
    for r in rows:
        for p in _safe_json(r.get("products"), []):
            t = (p.get("title") or "").strip() if isinstance(p, dict) else ""
            if t:
                titles.append(t)
        meta = _safe_json(r.get("page_meta"), {})
        for term in meta.get("search_term", []) if isinstance(meta, dict) else []:
            t = (term.get("term") or "").strip() if isinstance(term, dict) else ""
            if t:
                titles.append(t)
    seen, uniq = set(), []
    for t in titles:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append(t)
    if not uniq:
        return "I'm shopping on Amazon for an item I need."
    return f"I'm shopping on Amazon, looking for: {'; '.join(t[:80] for t in uniq[:3])}."


# --------------------------------------------------------------------------- #
# Session assembly + sliding-window record emission
# --------------------------------------------------------------------------- #
def _parse_step(row: dict) -> Step | None:
    payload = _assistant_payload(row)
    if payload is None:
        return None
    return Step(
        rationale=(row.get("rationale") or "").strip(),
        action=_flat_action(row) or {},
        assistant_payload=payload,
        # Preserve the complete observation. Sequence-length enforcement belongs
        # to the tokenizer/model pipeline (cfg.max_length), not dataset creation.
        html=row.get("simplified_html") or "",
    )

def _build_session(session_id: str, rows: list[dict], persona: str) -> Session | None:
    rows = sorted(rows, key=lambda r: r.get("timestamp") or "")
    steps: list[Step] = []
    for row in rows:
        # payload = _assistant_payload(row)
        step = _parse_step(row)
        if not step:
            continue
        steps.append(step)
    if not steps:
        return None
    return Session(session_id, persona, _intent_from_session(rows), steps)


def _load_opera(config: str, split: str):
    """Load OPeRA action + user + session tables for a given config/split."""
    from datasets import load_dataset

    logger.info("Loading OPeRA %s/%s (action, user, session) ...", config, split)
    action = load_dataset("NEU-HAI/OPeRA", f"{config}_action", split=split)
    user = load_dataset("NEU-HAI/OPeRA", f"{config}_user", split=split)
    session = load_dataset("NEU-HAI/OPeRA", f"{config}_session", split=split)
    user_by_id = {u["user_id"]: u for u in user}
    sess_user = {s["session_id"]: s["user_id"] for s in session}
    return action, user_by_id, sess_user


def _sessions_for_split(
    config: str, split: str, min_steps: int
) -> list[Session]:
    """Load one OPeRA split and build the valid (>= min_steps) sessions in it."""
    action, user_by_id, sess_user = _load_opera(config, split)

    by_session: dict[str, list[dict]] = {}
    for row in action:
        by_session.setdefault(row["session_id"], []).append(row)
    logger.info("Grouped %d %s actions into %d sessions", len(action), split, len(by_session))

    sessions: list[Session] = []
    for sid, rows in by_session.items():
        persona = _persona_from_user(user_by_id.get(sess_user.get(sid)))
        sess = _build_session(sid, rows, persona)
        if sess is not None and len(sess.steps) >= min_steps:
            sessions.append(sess)
    logger.info("Built %d valid %s sessions (>= %d steps)", len(sessions), split, min_steps)
    return sessions


def _push_to_hub(
    train_path: Path, val_path: Path, eval_path: Path, repo_id: str, private: bool = True
) -> None:
    """Publish the built SFT JSONL as an HF dataset with train/validation/test splits."""
    from datasets import DatasetDict, load_dataset

    dsdict = DatasetDict(
        {
            "train": load_dataset("json", data_files=str(train_path), split="train"),
            "validation": load_dataset("json", data_files=str(val_path), split="train"),
            "test": load_dataset("json", data_files=str(eval_path), split="train"),
        }
    )
    logger.info(
        "Pushing %d train / %d validation / %d test rows to hub repo '%s' (private=%s)",
        dsdict["train"].num_rows,
        dsdict["validation"].num_rows,
        dsdict["test"].num_rows,
        repo_id,
        private,
    )
    dsdict.push_to_hub(repo_id, private=private)
    logger.info("Pushed dataset to https://huggingface.co/datasets/%s", repo_id)


def build_sft_dataset(
    config: str = "filtered",
    output_dir: str = "data/opera",
    val_ratio: float = 0.1,
    window_size: int = 15,
    stride: int = 5,
    min_steps: int = 2,
    seed: int = 42,
    sample_size: int | None = None,
) -> tuple[str, str, str]:
    # Pull both OPeRA splits: `train` is carved into train/validation, `test` is used as-is.
    train_pool = _sessions_for_split(config, "train", min_steps)
    test_sessions = _sessions_for_split(config, "test", min_steps)
    if not train_pool:
        raise ValueError("No valid OPeRA train sessions produced.")
    if not test_sessions:
        raise ValueError("No valid OPeRA test sessions produced.")

    # Split the train pool at the session level so windows never span train/validation.
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    n_val = max(1, int(len(train_pool) * val_ratio))
    val_sessions, train_sessions = train_pool[:n_val], train_pool[n_val:]
    logger.info(
        "Split %d train-pool sessions -> %d train / %d validation; %d test sessions",
        len(train_pool), len(train_sessions), len(val_sessions), len(test_sessions),
    )

    # Cap the training sessions *after* the split, so validation and test are byte-identical
    # across sample sizes and low-real-data runs remain directly comparable.
    if sample_size is not None and sample_size > 0:
        if sample_size > len(train_sessions):
            raise ValueError(
                f"Requested --sample_size {sample_size} exceeds the {len(train_sessions)} "
                f"train session(s) available in OPeRA {config}."
            )
        train_sessions = rng.sample(train_sessions, sample_size)
        logger.info("Sampled %d train session(s)", sample_size)

    def _expand(sess: list[Session]) -> list[dict]:
        return [r for s in sess for r in session_to_records(s, window_size, stride)]

    train_records = _expand(train_sessions)
    val_records = _expand(val_sessions)
    eval_records = _expand(test_sessions)
    logger.info(
        "Built %d train / %d validation / %d test windows",
        len(train_records), len(val_records), len(eval_records),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "sft_train.jsonl"
    val_path = out / "sft_val.jsonl"
    eval_path = out / "sft_eval.jsonl"
    for path, recs in [(train_path, train_records), (val_path, val_records), (eval_path, eval_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d windows to %s", len(recs), path)
    return str(train_path), str(val_path), str(eval_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adapt OPeRA into Method 1 SFT JSONL.")
    p.add_argument("--config", choices=["filtered", "full"], default="filtered")
    p.add_argument("--output_dir", default="data/opera")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of the OPeRA train pool held out as validation.")
    p.add_argument("--window_size", type=int, default=15,
                   help="Steps per sliding window over a session.")
    p.add_argument("--stride", type=int, default=5,
                   help="Step advance between consecutive windows (overlap = window_size - stride).")
    p.add_argument("--min_steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_size", type=int, default=None,
                   help="Cap the number of training sessions kept (applied after the "
                        "validation split, so validation/test stay identical across sizes). "
                        "Omit or pass <= 0 to use all.")
    p.add_argument("--push_to_hub", action="store_true",
                   help="Push the built dataset to the HF Hub after writing JSONL.")
    p.add_argument("--repo_id", default=None,
                   help="Target HF dataset repo id (required with --push_to_hub).")
    p.add_argument("--private", dest="private", action="store_true", default=True,
                   help="Create/keep the hub repo private (default).")
    p.add_argument("--public", dest="private", action="store_false",
                   help="Make the hub repo public instead of private.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.push_to_hub and not args.repo_id:
        raise SystemExit("--repo_id is required when --push_to_hub is set.")
    train_path, val_path, eval_path = build_sft_dataset(
        config=args.config, output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        window_size=args.window_size, stride=args.stride,
        min_steps=args.min_steps, seed=args.seed,
        sample_size=args.sample_size,
    )
    print(f"Train: {train_path}  ({sum(1 for _ in open(train_path))} windows)")
    print(f"Val:   {val_path}  ({sum(1 for _ in open(val_path))} windows)")
    print(f"Eval:  {eval_path}  ({sum(1 for _ in open(eval_path))} windows)")
    if args.push_to_hub:
        _push_to_hub(Path(train_path), Path(val_path), Path(eval_path),
                     args.repo_id, private=args.private)
