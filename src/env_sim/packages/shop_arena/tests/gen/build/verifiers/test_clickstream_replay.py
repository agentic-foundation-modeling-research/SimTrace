"""Unit tests for :class:`ClickstreamReplayVerifier`.

Exercise the verifier through its two injection seams — a stub
:class:`~shop_arena.gen.final_eval.playwright_smoke.DevServerFactory`
yielding a fixed base URL and a stub ``TrajectoryReplayer`` returning
deterministic :class:`StepOutcome`\\ s — so no Chromium or Node toolchain
is required. Covers applicability (redo-suffix aware), the missing-data
skip (PASS), all-reproducible PASS, unreproducible FAIL feedback/details,
the missing-storefront FAIL, the retry-budget ADVISORY downgrade, and the
replay-crash ADVISORY.
"""

from __future__ import annotations

import contextlib
import csv
import json
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from harness.verifiers import Verdict, VerifierContext
from shop_arena.gen.build.verifiers._clickstream import Session
from shop_arena.gen.build.verifiers.clickstream_replay import (
    ClickstreamReplayVerifier,
    StepOutcome,
)
from shop_arena.gen.final_eval.playwright_smoke import DevServerFactory

_BASE_URL = "http://127.0.0.1:4321"

_HEADER: tuple[str, ...] = (
    "session_id",
    "timestamp",
    "semantic_action",
    "url",
    "product_handle",
)


# --------------------------------------------------------------------------- #
# Stub seams
# --------------------------------------------------------------------------- #


def _stub_factory(captured: list[Path] | None = None) -> DevServerFactory:
    """Build a :class:`DevServerFactory` yielding ``_BASE_URL``."""

    @contextlib.contextmanager
    def _server(hydrogen_dir: Path) -> Generator[str]:
        if captured is not None:
            captured.append(hydrogen_dir)
        yield _BASE_URL

    def _build(hydrogen_dir: Path) -> AbstractContextManager[str]:
        return _server(hydrogen_dir)

    return _build


def _never_booted_factory() -> DevServerFactory:
    """Build a :class:`DevServerFactory` that fails if it is ever entered."""

    def _build(hydrogen_dir: Path) -> AbstractContextManager[str]:
        del hydrogen_dir
        raise AssertionError("dev server must not boot")

    return _build


class _StubReplayer:
    """Records its call and returns preset outcomes (or raises)."""

    def __init__(
        self,
        outcomes: Sequence[StepOutcome] = (),
        *,
        raises: Exception | None = None,
    ) -> None:
        self._outcomes = tuple(outcomes)
        self._raises = raises
        self.calls: list[tuple[str, tuple[Session, ...], float]] = []

    def __call__(
        self,
        *,
        base_url: str,
        sessions: Sequence[Session],
        timeout_s: float,
    ) -> tuple[StepOutcome, ...]:
        self.calls.append((base_url, tuple(sessions), timeout_s))
        if self._raises is not None:
            raise self._raises
        return self._outcomes


def _outcome(
    *,
    session_id: str = "s1",
    step_index: int = 0,
    action: str = "detail",
    url: str | None = f"{_BASE_URL}/products/x",
    reproducible: bool = True,
    skipped: bool = False,
    reason: str = "",
) -> StepOutcome:
    """Build a :class:`StepOutcome` for the stub replayer."""
    return StepOutcome(
        session_id=session_id,
        step_index=step_index,
        action=action,
        url=url,
        reproducible=reproducible,
        skipped=skipped,
        reason=reason,
    )


def _write_csv(path: Path, session_ids: Sequence[str]) -> Path:
    """Write a minimal one-detail-step-per-session clickstream CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_HEADER))
        writer.writeheader()
        for sid in session_ids:
            writer.writerow(
                {
                    "session_id": sid,
                    "timestamp": "1",
                    "semantic_action": "detail",
                    "url": "/products/x",
                    "product_handle": "x",
                },
            )
    return path


def _write_prior_fail(*, run_dir: Path, iter_id: str, task_id: str) -> None:
    """Write a dispatch-shaped FAIL telemetry sibling for the retry-budget scan."""
    record_dir = run_dir / "iters" / iter_id / "checks" / "verifiers"
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "clickstream_replay.json").write_text(
        json.dumps(
            {
                "iter_id": iter_id,
                "name": "clickstream_replay",
                "task_id": task_id,
                "verdict": "fail",
                "details": {},
            },
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Identity / applicability
# --------------------------------------------------------------------------- #


def test_name_matches_spec(tmp_path: Path) -> None:
    verifier = ClickstreamReplayVerifier(
        clickstream_path=tmp_path / "clickstream.csv",
        dev_server_factory=_stub_factory(),
        replayer=_StubReplayer(),
    )
    assert verifier.name == "clickstream_replay"


def test_applies_to_default_tasks(tmp_path: Path) -> None:
    verifier = ClickstreamReplayVerifier(
        clickstream_path=tmp_path / "clickstream.csv",
        dev_server_factory=_stub_factory(),
        replayer=_StubReplayer(),
    )
    assert verifier.applies_to("gen_navigation") is True
    assert verifier.applies_to("visual_fix") is True
    assert verifier.applies_to("gen_homepage") is False
    assert verifier.applies_to("plan") is False


def test_applies_to_strips_redo_suffix(tmp_path: Path) -> None:
    verifier = ClickstreamReplayVerifier(
        clickstream_path=tmp_path / "clickstream.csv",
        dev_server_factory=_stub_factory(),
        replayer=_StubReplayer(),
    )
    assert verifier.applies_to("visual_fix_redo_2") is True
    assert verifier.applies_to("gen_navigation_redo_10") is True


def test_applies_to_honours_custom_task_set(tmp_path: Path) -> None:
    verifier = ClickstreamReplayVerifier(
        clickstream_path=tmp_path / "clickstream.csv",
        dev_server_factory=_stub_factory(),
        replayer=_StubReplayer(),
        applicable_tasks={"gen_homepage"},
    )
    assert verifier.applies_to("gen_homepage") is True
    assert verifier.applies_to("visual_fix") is False


# --------------------------------------------------------------------------- #
# Missing-data skip → PASS
# --------------------------------------------------------------------------- #


def test_run_skips_pass_when_no_sessions(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """An empty CSV (no sessions) surfaces as a PASS skip; nothing boots."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", [])
    replayer = _StubReplayer()
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_never_booted_factory(),
        replayer=replayer,
    )

    result = verifier.run(make_ctx(selected_task_id="gen_navigation"))

    assert result.verdict is Verdict.PASS
    assert result.details["ran"] is False
    assert result.details["skipped"] is True
    assert replayer.calls == []


def test_run_advisory_when_csv_missing(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """An unreadable CSV path downgrades to ADVISORY (does not wedge the loop)."""
    verifier = ClickstreamReplayVerifier(
        clickstream_path=tmp_path / "does-not-exist.csv",
        dev_server_factory=_never_booted_factory(),
        replayer=_StubReplayer(),
    )

    result = verifier.run(make_ctx(selected_task_id="gen_navigation"))

    assert result.verdict is Verdict.ADVISORY
    assert result.details["skipped"] is True


# --------------------------------------------------------------------------- #
# Replay → verdict
# --------------------------------------------------------------------------- #


def test_run_passes_when_every_step_reproducible(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """Every non-skipped step reproducible → PASS with tallied details."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1", "s2"])
    captured: list[Path] = []
    replayer = _StubReplayer(
        [
            _outcome(session_id="s1", step_index=0),
            _outcome(session_id="s2", step_index=0),
        ],
    )
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_stub_factory(captured),
        replayer=replayer,
    )

    result = verifier.run(make_ctx(selected_task_id="gen_navigation"))

    assert result.verdict is Verdict.PASS
    assert result.details["ran"] is True
    assert result.details["sessions"] == 2
    assert result.details["steps_total"] == 2
    assert result.details["doable"] == 2
    assert result.details["not_doable"] == 0
    # The dev server was booted against the storefront tree.
    assert captured and captured[0].name == "hydrogen"
    # The replayer saw the base URL and both sampled sessions.
    (base_url, sessions, _timeout) = replayer.calls[0]
    assert base_url == _BASE_URL
    assert {s.session_id for s in sessions} == {"s1", "s2"}


def test_run_fails_when_a_step_not_reproducible(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """A single unreproducible step FAILs with a feedback line naming it."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1"])
    replayer = _StubReplayer(
        [
            _outcome(session_id="s1", step_index=0, reproducible=True),
            _outcome(
                session_id="s1",
                step_index=1,
                action="add",
                url=f"{_BASE_URL}/products/ghost",
                reproducible=False,
                reason="no add-to-cart control found on the page",
            ),
        ],
    )
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_stub_factory(),
        replayer=replayer,
    )

    result = verifier.run(make_ctx(selected_task_id="visual_fix"))

    assert result.verdict is Verdict.FAIL
    assert result.details["not_doable"] == 1
    assert result.details["doable"] == 1
    (failure,) = result.details["failures"]
    assert failure["session_id"] == "s1"
    assert failure["step"] == 1
    assert failure["action"] == "add"
    assert "no add-to-cart control" in failure["reason"]
    assert "s1" in result.feedback
    assert "add-to-cart" in result.feedback


def test_run_skipped_steps_do_not_gate_verdict(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """Skipped steps are advisory: PASS even though they are not 'doable'."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1"])
    replayer = _StubReplayer(
        [
            _outcome(session_id="s1", step_index=0, reproducible=True),
            _outcome(
                session_id="s1",
                step_index=1,
                action="mystery",
                url=None,
                skipped=True,
                reason="action `mystery` carried no url",
            ),
        ],
    )
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_stub_factory(),
        replayer=replayer,
    )

    result = verifier.run(make_ctx(selected_task_id="gen_navigation"))

    assert result.verdict is Verdict.PASS
    assert result.details["steps_total"] == 1
    assert result.details["steps_skipped"] == 1


# --------------------------------------------------------------------------- #
# Missing storefront tree → FAIL
# --------------------------------------------------------------------------- #


def test_run_fails_when_storefront_tree_missing(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """A missing ``app_dir`` under the artifact dir is a FAIL (not skip)."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1"])
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_never_booted_factory(),
        replayer=_StubReplayer(),
        app_dir=Path("nonexistent"),
    )

    result = verifier.run(make_ctx(selected_task_id="gen_navigation"))

    assert result.verdict is Verdict.FAIL
    assert result.details["exists"] is False


# --------------------------------------------------------------------------- #
# Retry budget → ADVISORY
# --------------------------------------------------------------------------- #


def test_run_downgrades_to_advisory_when_retry_budget_met(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """After ``retry_budget`` sibling FAILs the verifier downgrades without booting."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1"])
    ctx = make_ctx(selected_task_id="gen_navigation", iter_id="exec-0004")
    for i in range(1, 4):
        _write_prior_fail(run_dir=ctx.run_dir, iter_id=f"exec-{i:04d}", task_id="gen_navigation")
    replayer = _StubReplayer()
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_never_booted_factory(),
        replayer=replayer,
        retry_budget=3,
    )

    result = verifier.run(ctx)

    assert result.verdict is Verdict.ADVISORY
    assert result.details["retry_budget_exhausted"] is True
    assert result.details["prior_fails"] == 3
    assert replayer.calls == []


def test_run_does_not_downgrade_below_retry_budget(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """With fewer prior FAILs than the budget the verifier still replays."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1"])
    ctx = make_ctx(selected_task_id="gen_navigation", iter_id="exec-0003")
    for i in range(1, 3):
        _write_prior_fail(run_dir=ctx.run_dir, iter_id=f"exec-{i:04d}", task_id="gen_navigation")
    replayer = _StubReplayer([_outcome(session_id="s1")])
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_stub_factory(),
        replayer=replayer,
        retry_budget=3,
    )

    result = verifier.run(ctx)

    assert result.verdict is Verdict.PASS
    assert result.details["prior_fails"] == 2
    assert len(replayer.calls) == 1


# --------------------------------------------------------------------------- #
# Replay crash → ADVISORY
# --------------------------------------------------------------------------- #


def test_run_advisory_when_replay_raises(
    make_ctx: Callable[..., VerifierContext],
    tmp_path: Path,
) -> None:
    """A replay/dev-server crash surfaces as ADVISORY, never FAIL."""
    csv_path = _write_csv(tmp_path / "clickstream.csv", ["s1"])
    replayer = _StubReplayer(raises=RuntimeError("chromium missing"))
    verifier = ClickstreamReplayVerifier(
        clickstream_path=csv_path,
        dev_server_factory=_stub_factory(),
        replayer=replayer,
    )

    result = verifier.run(make_ctx(selected_task_id="gen_navigation"))

    assert result.verdict is Verdict.ADVISORY
    assert result.details["skipped"] is True
    assert "chromium missing" in result.details["error"]
