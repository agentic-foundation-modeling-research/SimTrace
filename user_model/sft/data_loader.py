"""Load simulation output directories and yield per-step records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from user_model.prompt_templates import normalize_assistant_payload, render_system_prompt

logger = logging.getLogger(__name__)


@dataclass
class Step:
    rationale: str
    action: dict
    assistant_payload: str  # canonical {"rationale": ..., "action": ...} JSON string
    html: str


@dataclass
class Session:
    session_id: str
    persona: str
    intent: str
    steps: list[Step]

def session_to_records(session: Session, window_size: int, stride: int) -> list[dict]:
    """Emit one multi-turn record per sliding window over the session's steps."""
    system_msg = {"role": "system", "content": render_system_prompt(session.persona, session.intent)}
    records: list[dict] = []
    for win_idx, (start, end) in enumerate(window_bounds(len(session.steps), window_size, stride)):
        messages: list[dict] = [system_msg]
        for step in session.steps[start:end]:
            messages.append({"role": "user", "content": f"# context\n{step.html}"})
            messages.append({"role": "assistant", "content": step.assistant_payload})
        records.append({"session_id": f"{session.session_id}#w{win_idx}", "messages": messages})
    return records


def _read_html(path: str | None, run_dir: Path | None = None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute() and run_dir is not None:
        p = run_dir / p
    if not p.exists():
        logger.warning("HTML file not found: %s", path)
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def _parse_step(record: dict, run_dir: Path | None = None) -> Step | None:
    payload = normalize_assistant_payload(record.get("synthetic_action", ""))
    if not payload:
        return None
    
    rationale = payload.get("rationale", "")
    action = {k: v for k, v in payload.items() if k != "rationale"}
    return Step(
        rationale=rationale,
        action=action,
        assistant_payload=json.dumps(payload, ensure_ascii=False),
        html=_read_html(record.get("dom_snapshot_simplified"), run_dir),
    )


def _post_verify_avg(run_dir: Path) -> float | None:
    """Mean of the numeric ``score`` fields in ``post_verify_result.json``.

    Returns ``None`` when the file is absent or has no scorable dict-valued
    entries (callers treat ``None`` as "keep"). Mirrors the extraction in
    ``eval/run_eval.py::_compute_post_verify_exclusions`` so training and eval
    apply the same quality gate.
    """
    verify_path = run_dir / "post_verify_result.json"
    if not verify_path.is_file():
        return None
    try:
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse post_verify_result.json in %s: %s", run_dir, exc)
        return None
    scores = [
        v["score"]
        for v in verify.values()
        if isinstance(v, dict) and isinstance(v.get("score"), (int, float))
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _load_session(
    run_dir: Path, min_steps: int, post_verify_threshold: float | None = None
) -> Session | None:
    session_path = run_dir / "session_data.json"
    if not session_path.exists() or (run_dir / "error.txt").exists():
        return None
    if post_verify_threshold is not None:
        avg = _post_verify_avg(run_dir)
        if avg is not None and avg < post_verify_threshold:
            logger.info(
                "Dropping %s: post-verify avg %.3f < threshold %.3f",
                run_dir.name, avg, post_verify_threshold,
            )
            return None
    try:
        records: list[dict] = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse session_data.json in %s: %s", run_dir, exc)
        return None
    if len(records) < min_steps:
        return None
    steps = [s for s in (_parse_step(r, run_dir) for r in records) if s is not None]
    if len(steps) < min_steps:
        return None

    info_path = run_dir / "basic_info.json"
    basic_info: dict = {}
    if info_path.exists():
        try:
            basic_info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse basic_info.json in %s: %s", run_dir, exc)

    return Session(
        session_id=records[0].get("session_id", run_dir.name),
        persona=basic_info.get("persona", ""),
        intent=basic_info.get("intent", ""),
        steps=steps,
    )


def iter_sessions(
    run_dir: str | Path,
    min_steps: int = 2,
    post_verify_threshold: float | None = None,
) -> Iterator[Session]:
    """Yield each valid Session under <run_dir>/runs/.

    When ``post_verify_threshold`` is set, sessions whose
    ``post_verify_result.json`` average score falls below it are dropped;
    sessions without a verifier file are kept.
    """
    run_dir = Path(run_dir)
    runs_root = run_dir / "runs"
    if not runs_root.exists():
        runs_root = run_dir
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        session = _load_session(d, min_steps, post_verify_threshold)
        if session is not None:
            yield session


def window_bounds(n: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    """Return (start, end) step ranges for fixed-step sliding windows over n steps.

    Windows advance by ``stride``; the final window is anchored to the end so no
    trailing steps are dropped. Overlap between consecutive windows is
    ``window_size - stride``.
    """
    if n <= window_size:
        return [(0, n)]
    starts = list(range(0, n - window_size + 1, stride))
    if starts[-1] + window_size < n:
        starts.append(n - window_size)  # anchor a full-size final window
    return [(s, s + window_size) for s in starts]
