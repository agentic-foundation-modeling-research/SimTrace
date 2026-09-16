"""Cross-iteration rollup of ``clickstream_replay`` telemetry (final_eval).

The ``clickstream_replay`` verifier writes a per-iteration telemetry file
(``runs/build/iters/<iter_id>/checks/verifiers/clickstream_replay.json``)
after every applicable iteration, storing its ``details`` verbatim (the
harness dispatch layer owns that write). This module tallies those files
into a single ``clickstream`` subtree that the ``final_eval`` step merges
into ``<out_dir>/final_eval.json`` — the same pattern the visual sweep
uses for its ``visual`` subtree.

The subtree is **advisory**: it summarizes what the verifier already
recorded per iteration and never gates the run.

Module is import-safe: no I/O, no env reads, no side effects at import.
The single filesystem read happens inside :func:`build_clickstream_subtree`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

_RECORD_GLOB: Final[str] = "exec-*/checks/verifiers/clickstream_replay.json"
"""Glob (relative to ``runs/build/iters/``) of per-iteration telemetry."""


@dataclass(frozen=True, slots=True)
class _IterRecord:
    """One iteration's parsed clickstream telemetry (a ``ran`` record)."""

    iter_id: str
    verdict: str
    sessions: int
    dropped: int
    steps_total: int
    doable: int
    not_doable: int


def build_clickstream_subtree(iters_dir: Path) -> dict[str, Any]:
    """Roll up per-iteration ``clickstream_replay`` telemetry into a subtree.

    Scans ``<iters_dir>/exec-*/checks/verifiers/clickstream_replay.json``,
    keeps the records where the verifier actually replayed sessions
    (``details.ran == true``; skip records carry no counts), and tallies
    them into the advisory ``clickstream`` subtree.

    Args:
        iters_dir: The build loop's ``runs/build/iters/`` directory.
            A missing directory yields a ``ran: false`` subtree.

    Returns:
        A JSON-serialisable mapping:

        * ``ran`` — ``true`` when at least one iteration replayed
          sessions; ``false`` otherwise (no CSV, verifier never ran, or
          every run was a skip).
        * ``sessions_sampled`` / ``dropped`` — from the final replaying
          iteration (``0`` when none ran).
        * ``iterations`` — one row per replaying iteration, ordered by
          ``iter_id``: ``{iter_id, verdict, steps_total, doable,
          not_doable}``.
        * ``final`` — the last replaying iteration's
          ``{verdict, steps_total, doable, not_doable}``, or ``null``
          when none ran.
    """
    records = _collect_records(iters_dir)
    if not records:
        return {
            "ran": False,
            "sessions_sampled": 0,
            "dropped": 0,
            "iterations": [],
            "final": None,
        }
    final = records[-1]
    return {
        "ran": True,
        "sessions_sampled": final.sessions,
        "dropped": final.dropped,
        "iterations": [
            {
                "iter_id": record.iter_id,
                "verdict": record.verdict,
                "steps_total": record.steps_total,
                "doable": record.doable,
                "not_doable": record.not_doable,
            }
            for record in records
        ],
        "final": {
            "verdict": final.verdict,
            "steps_total": final.steps_total,
            "doable": final.doable,
            "not_doable": final.not_doable,
        },
    }


def _collect_records(iters_dir: Path) -> list[_IterRecord]:
    """Parse every replaying telemetry file, ordered by ``iter_id``."""
    if not iters_dir.is_dir():
        return []
    records: list[_IterRecord] = []
    for path in sorted(iters_dir.glob(_RECORD_GLOB)):
        record = _parse_record(path)
        if record is not None:
            records.append(record)
    return records


def _parse_record(path: Path) -> _IterRecord | None:
    """Parse one telemetry file into an :class:`_IterRecord`, or ``None``.

    Returns ``None`` for unreadable / malformed files and for skip
    records (``details.ran`` falsy) — a skip carries no doable /
    not-doable counts to aggregate.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    record = cast("dict[str, Any]", payload)
    details = record.get("details")
    if not isinstance(details, dict):
        return None
    details_dict = cast("dict[str, Any]", details)
    if not details_dict.get("ran"):
        return None
    iter_id = record.get("iter_id")
    verdict = record.get("verdict")
    if not isinstance(iter_id, str) or not isinstance(verdict, str):
        return None
    return _IterRecord(
        iter_id=iter_id,
        verdict=verdict,
        sessions=_as_int(details_dict.get("sessions")),
        dropped=_as_int(details_dict.get("dropped")),
        steps_total=_as_int(details_dict.get("steps_total")),
        doable=_as_int(details_dict.get("doable")),
        not_doable=_as_int(details_dict.get("not_doable")),
    )


def _as_int(value: Any) -> int:
    """Coerce a telemetry count to ``int``; non-ints (incl. bools) → ``0``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = ["build_clickstream_subtree"]
