"""Unit tests for :func:`build_clickstream_subtree` (final_eval rollup).

The verifier writes one telemetry file per applicable iteration
(``runs/build/iters/<iter_id>/checks/verifiers/clickstream_replay.json``,
storing its ``details`` verbatim). This module tallies those files into
the advisory ``clickstream`` subtree merged into ``final_eval.json``.
Tests write fixture iter dirs and assert the resulting subtree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shop_arena.gen.final_eval.clickstream import build_clickstream_subtree


def _write_record(
    iters_dir: Path,
    iter_id: str,
    *,
    verdict: str,
    details: dict[str, Any],
) -> None:
    """Write a dispatch-shaped ``clickstream_replay.json`` under ``iter_id``."""
    record_dir = iters_dir / iter_id / "checks" / "verifiers"
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "clickstream_replay.json").write_text(
        json.dumps(
            {
                "iter_id": iter_id,
                "name": "clickstream_replay",
                "task_id": "gen_navigation",
                "verdict": verdict,
                "details": details,
            },
        ),
        encoding="utf-8",
    )


def _ran_details(*, sessions: int, dropped: int, steps: int, doable: int) -> dict[str, Any]:
    """Build a ``ran`` details payload (as the verifier writes it)."""
    return {
        "ran": True,
        "sessions": sessions,
        "dropped": dropped,
        "steps_total": steps,
        "doable": doable,
        "not_doable": steps - doable,
    }


# --------------------------------------------------------------------------- #
# Empty / missing
# --------------------------------------------------------------------------- #


def test_missing_iters_dir_yields_not_ran(tmp_path: Path) -> None:
    """A missing ``iters/`` directory yields a ``ran: false`` subtree."""
    subtree = build_clickstream_subtree(tmp_path / "runs" / "build" / "iters")

    assert subtree == {
        "ran": False,
        "sessions_sampled": 0,
        "dropped": 0,
        "iterations": [],
        "final": None,
    }


def test_only_skip_records_yields_not_ran(tmp_path: Path) -> None:
    """Skip records (``ran`` falsy) carry no counts and are ignored."""
    iters = tmp_path / "iters"
    _write_record(iters, "exec-0001", verdict="pass", details={"ran": False, "skipped": True})

    subtree = build_clickstream_subtree(iters)

    assert subtree["ran"] is False
    assert subtree["iterations"] == []


# --------------------------------------------------------------------------- #
# Rollup
# --------------------------------------------------------------------------- #


def test_rollup_orders_iterations_and_picks_final(tmp_path: Path) -> None:
    """Replaying iterations are ordered by id; ``final`` is the last one."""
    iters = tmp_path / "iters"
    _write_record(
        iters,
        "exec-0007",
        verdict="fail",
        details=_ran_details(sessions=15, dropped=2, steps=42, doable=39),
    )
    _write_record(
        iters,
        "exec-0009",
        verdict="pass",
        details=_ran_details(sessions=15, dropped=2, steps=42, doable=42),
    )

    subtree = build_clickstream_subtree(iters)

    assert subtree["ran"] is True
    assert subtree["sessions_sampled"] == 15
    assert subtree["dropped"] == 2
    assert [it["iter_id"] for it in subtree["iterations"]] == ["exec-0007", "exec-0009"]
    assert subtree["iterations"][0] == {
        "iter_id": "exec-0007",
        "verdict": "fail",
        "steps_total": 42,
        "doable": 39,
        "not_doable": 3,
    }
    assert subtree["final"] == {
        "verdict": "pass",
        "steps_total": 42,
        "doable": 42,
        "not_doable": 0,
    }


def test_rollup_ignores_skip_records_among_ran(tmp_path: Path) -> None:
    """A skip record between ran records is dropped from the rollup."""
    iters = tmp_path / "iters"
    _write_record(iters, "exec-0001", verdict="pass", details={"ran": False, "skipped": True})
    _write_record(
        iters,
        "exec-0002",
        verdict="pass",
        details=_ran_details(sessions=3, dropped=0, steps=5, doable=5),
    )

    subtree = build_clickstream_subtree(iters)

    assert [it["iter_id"] for it in subtree["iterations"]] == ["exec-0002"]
    assert subtree["final"]["verdict"] == "pass"


def test_rollup_skips_malformed_records(tmp_path: Path) -> None:
    """Unreadable / malformed telemetry files are silently ignored."""
    iters = tmp_path / "iters"
    bad_dir = iters / "exec-0001" / "checks" / "verifiers"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "clickstream_replay.json").write_text("{not json", encoding="utf-8")
    _write_record(
        iters,
        "exec-0002",
        verdict="pass",
        details=_ran_details(sessions=1, dropped=0, steps=1, doable=1),
    )

    subtree = build_clickstream_subtree(iters)

    assert [it["iter_id"] for it in subtree["iterations"]] == ["exec-0002"]
