"""``clickstream_replay`` build-loop verifier.

Follows real user **clickstream** trajectories and checks whether each
session can be *strictly reproduced* on the generated **twin**. It is in
the same family as the navigation verifiers (``nav_coverage``,
``routes_200``): where ``routes_200`` probes a *sampled* set of bucket
routes, this verifier exercises the *real* paths users walked on the
live store, mapped onto the twin.

The verifier never contacts the live store. Each trajectory step carries
a path (its ``url`` column); that path is joined onto the twin's
dev-server base URL and driven with Playwright to confirm the concrete
action is reachable:

* ``detail`` — navigate to the product-detail page; reachable iff the
  page responds ``< 400`` and renders a product (not the 404 boundary).
* ``add`` — on the product page, submit the add-to-cart form (which
  POSTs to ``/cart`` with ``intent=add``), then confirm the ``/cart``
  page reflects the line.
* ``explore-stay`` — a scroll/browse no-op; reachable if a page is
  loaded.
* ``terminate`` — session end; always reproducible.
* any other action carrying a path — generic navigation; reachable iff
  the response is ``< 400``.
* an unknown action with no path — recorded as *skipped* (advisory).

**Twin mode.** Strict reproduction only makes sense in twin
(``catalog_source="ingest"``) mode, where the twin preserves the live
store's product/collection handles and URL paths. In ``synth`` mode the
handles differ and most ``detail``/``add`` steps fail — hence the
verifier is opt-in (registered only when a clickstream CSV is supplied).

**Missing data is not a failure.** No matching sessions (empty CSV, or a
``store_id`` filter that excludes everything) surfaces as ``PASS`` with
an explanatory note. Only a genuinely non-reproducible step FAILs.

The browser automation lives behind the :class:`TrajectoryReplayer`
seam: production wiring is :class:`PlaywrightTrajectoryReplayer` (sync
Playwright, one browser, a fresh context per session so carts do not
leak); tests inject a stub that returns deterministic per-step outcomes
so the verifier is exercised without launching Chromium.

The add-to-cart selector heuristics (:data:`_ADD_TO_CART_SELECTOR`) are
the one brittle point — they must track the storefront template's
product form. They are module constants so a template change is a
one-line edit.

Module is import-safe: no I/O, no env reads, no side effects at import.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import playwright.sync_api

from harness.verifiers import Verdict, VerifierContext, VerifierResult
from shop_arena.gen.build.verifiers._clickstream import (
    ACTION_ADD,
    ACTION_DETAIL,
    ACTION_EXPLORE_STAY,
    ACTION_TERMINATE,
    DEFAULT_MAX_SESSIONS,
    ClickstreamEvent,
    Session,
    load_sessions,
)
from shop_arena.gen.build.verifiers._history import count_prior_task_fails
from shop_arena.gen.final_eval.playwright_smoke import DevServerFactory

_log = logging.getLogger(__name__)

_NAME: Final[str] = "clickstream_replay"
"""Verifier name (filesystem-safe; the per-verifier telemetry stem)."""

_HYDROGEN_DIRNAME: Final[str] = "hydrogen"
"""Default storefront app dir relative to ``VerifierContext.artifact_dir``."""

_DEFAULT_RETRY_BUDGET: Final[int] = 3
"""Per-task cap on consecutive FAILs before downgrading to ADVISORY."""

_DEFAULT_TIMEOUT_S: Final[float] = 30.0
"""Per-navigation Playwright budget (seconds)."""

_DEFAULT_TASKS: Final[frozenset[str]] = frozenset({"gen_navigation", "visual_fix"})
"""Tasks this verifier gates by default (spec + confirmed scope)."""

_REDO_SUFFIX: Final[re.Pattern[str]] = re.compile(r"_redo_\d+$")
"""Trailing ``_redo_<n>`` suffix; stripped before applicability lookup."""

_ADD_TO_CART_SELECTOR: Final[str] = (
    "form[action='/cart'] button[type='submit'], "
    "button[name='add'], "
    "button[data-testid='add-to-cart']"
)
"""Selector list the replayer clicks to add the current product to the cart.

Tracks the storefront template's product form (see the react-vite
``routes/product.tsx`` variant form, which POSTs to ``/cart`` with
``intent=add``). Brittle by nature — update alongside the template.
"""

_VARIANT_TRIGGER_SELECTOR: Final[str] = (
    "form[action='/cart'] [role='combobox'], form[action='/cart'] .pdp-size-trigger"
)
"""Opens the product's variant/size picker when add-to-cart starts disabled.

Templates commonly gate the add button behind a required size selection
(the react-vite ``VariantSelector`` disables the submit until a size is
chosen). Clicking this trigger reveals the options listbox. Brittle by
nature — update alongside the template.
"""

_VARIANT_OPTION_SELECTOR: Final[str] = (
    "form[action='/cart'] [role='option'] button:not([disabled]), "
    "form[action='/cart'] .pdp-size-option:not([disabled])"
)
"""Selects the first in-stock variant option, enabling add-to-cart."""

_PRODUCT_HEADING_SELECTOR: Final[str] = "#product-heading, h1"
"""Signal that a product-detail page rendered (not the error boundary)."""

_HTTP_ERROR_FLOOR: Final[int] = 400
"""Status codes ``>=`` this mean the navigation did not reach a real page."""


# --------------------------------------------------------------------------- #
# Per-step outcome + replayer seam
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Result of replaying one trajectory step on the twin.

    Attributes:
        session_id: Session the step belongs to.
        step_index: 0-based index of the step within the session.
        action: The step's ``semantic_action``.
        url: The mapped twin URL that was exercised, or ``None`` when the
            step carried no path.
        reproducible: Whether the action succeeded on the twin.
        skipped: Whether the step was skipped (unknown action with no
            path) — skipped steps are advisory and never FAIL the run.
        reason: Human-readable explanation (empty on a clean success).
    """

    session_id: str
    step_index: int
    action: str
    url: str | None
    reproducible: bool
    skipped: bool
    reason: str


@runtime_checkable
class TrajectoryReplayer(Protocol):
    """Replays sessions against ``base_url`` and reports per-step outcomes.

    Production wiring is :class:`PlaywrightTrajectoryReplayer`; tests
    inject a stub returning deterministic :class:`StepOutcome`\\ s so the
    verifier runs without a browser or dev server.
    """

    def __call__(
        self,
        *,
        base_url: str,
        sessions: Sequence[Session],
        timeout_s: float,
    ) -> tuple[StepOutcome, ...]:
        """Replay every step of every session and return the outcomes.

        Implementations should isolate sessions from one another (a fresh
        browser context per session) so a cart added in one session does
        not leak into the next.
        """
        ...


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


class ClickstreamReplayVerifier:
    """Gates a task on strict reproduction of real user trajectories.

    Attributes:
        name: ``"clickstream_replay"`` — the per-verifier telemetry
            filename and the ``feedback.md`` section heading.
    """

    name: str = _NAME

    def __init__(
        self,
        *,
        clickstream_path: Path,
        dev_server_factory: DevServerFactory,
        replayer: TrajectoryReplayer | None = None,
        app_dir: Path = Path(_HYDROGEN_DIRNAME),
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        store_id: str | None = None,
        retry_budget: int = _DEFAULT_RETRY_BUDGET,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        applicable_tasks: Iterable[str] | None = None,
    ) -> None:
        """Build the verifier with optional injection seams.

        Args:
            clickstream_path: Path to the clickstream CSV to replay.
            dev_server_factory: Boots the twin dev server; production
                wiring is the ``pnpm dev`` runner, tests inject a stub
                yielding a deterministic base URL.
            replayer: Trajectory replayer. Defaults to
                :class:`PlaywrightTrajectoryReplayer` (sync Playwright).
            app_dir: Storefront app directory relative to
                ``VerifierContext.artifact_dir``. Defaults to ``hydrogen``.
            max_sessions: Cap on sampled sessions (deterministic: first N
                by sorted ``session_id``). Non-positive disables the cap.
            store_id: When set, replay only rows for this store.
            retry_budget: Per-task cap on consecutive FAILs before the
                verifier downgrades to ADVISORY. ``0`` disables it.
            timeout_s: Per-navigation Playwright budget (seconds).
            applicable_tasks: Override the default applicability set.
                Defaults to :data:`_DEFAULT_TASKS`.
        """
        self._clickstream_path = clickstream_path
        self._dev_server_factory = dev_server_factory
        self._replayer: TrajectoryReplayer = (
            replayer if replayer is not None else PlaywrightTrajectoryReplayer()
        )
        self._app_dir = app_dir
        self._max_sessions = max_sessions
        self._store_id = store_id
        self._retry_budget = retry_budget
        self._timeout_s = timeout_s
        self._applicable_tasks: frozenset[str] = (
            _DEFAULT_TASKS if applicable_tasks is None else frozenset(applicable_tasks)
        )

    def applies_to(self, task_id: str) -> bool:
        """Match the configured applicability set (redo-suffix aware).

        Args:
            task_id: Selected task id.

        Returns:
            ``True`` when this verifier should run for ``task_id``. A
            trailing ``_redo_<n>`` suffix is stripped first so redo task
            ids reuse the base task's scope.
        """
        base = _REDO_SUFFIX.sub("", task_id)
        return base in self._applicable_tasks

    def run(self, ctx: VerifierContext) -> VerifierResult:
        """Replay the sampled sessions on the twin and map the verdict.

        Args:
            ctx: Verifier context. Reads ``ctx.artifact_dir`` for the
                storefront tree and ``ctx.run_dir`` / ``ctx.iter_id`` for
                the retry-budget scan.

        Returns:
            ``PASS`` when every non-skipped step reproduced (or when
            there are no sessions to replay — the skip case). ``FAIL``
            (with per-step feedback) when any step is not reproducible on
            the twin. A CSV that cannot be read, an exhausted retry
            budget, or a replay/dev-server crash surface as ``ADVISORY``
            so the loop proceeds. A missing storefront tree is ``FAIL``.
        """
        # Step 1: parse + sample sessions.
        try:
            sampled = load_sessions(
                self._clickstream_path,
                store_id=self._store_id,
                max_sessions=self._max_sessions,
            )
        except OSError as exc:
            return VerifierResult(
                verdict=Verdict.ADVISORY,
                feedback=(
                    f"`clickstream_replay` could not read the clickstream CSV at "
                    f"`{self._clickstream_path}`: {exc}. Skipping."
                ),
                details={"clickstream_path": str(self._clickstream_path), "skipped": True},
            )
        if not sampled.sessions:
            return VerifierResult(
                verdict=Verdict.PASS,
                feedback=(
                    "`clickstream_replay` found no sessions to replay "
                    f"(store_id={self._store_id!r}); skipping."
                ),
                details={
                    "ran": False,
                    "sessions": 0,
                    "dropped": sampled.dropped,
                    "skipped": True,
                },
            )
        _log.info(
            "clickstream_replay task=%s sessions=%d dropped=%d",
            ctx.selected_task_id,
            len(sampled.sessions),
            sampled.dropped,
        )

        # Step 2: per-task retry budget — break the loop before booting.
        prior_fails = count_prior_task_fails(
            run_dir=ctx.run_dir,
            iter_id=ctx.iter_id,
            verifier_name=self.name,
            task_id=ctx.selected_task_id,
        )
        if self._retry_budget > 0 and prior_fails >= self._retry_budget:
            return VerifierResult(
                verdict=Verdict.ADVISORY,
                feedback=(
                    f"`clickstream_replay` has FAILed {prior_fails} time(s) against "
                    f"task `{ctx.selected_task_id}`, meeting the retry budget "
                    f"({self._retry_budget}). Downgrading to ADVISORY."
                ),
                details={
                    "task_id": ctx.selected_task_id,
                    "retry_budget": self._retry_budget,
                    "retry_budget_exhausted": True,
                    "prior_fails": prior_fails,
                },
            )

        # Step 3: resolve the storefront tree.
        app_dir = ctx.artifact_dir / self._app_dir
        if not app_dir.is_dir():
            return VerifierResult(
                verdict=Verdict.FAIL,
                feedback=(
                    f"`clickstream_replay` could not find the storefront tree at "
                    f"`{self._app_dir.as_posix()}/`. Did `clone_template` run?"
                ),
                details={"app_dir": str(app_dir), "exists": False},
            )

        # Step 4: boot the dev server + replay.
        try:
            with self._dev_server_factory(app_dir) as base_url:
                outcomes = self._replayer(
                    base_url=base_url,
                    sessions=sampled.sessions,
                    timeout_s=self._timeout_s,
                )
        except Exception as exc:
            _log.warning("clickstream_replay: replay raised: %s", exc)
            return VerifierResult(
                verdict=Verdict.ADVISORY,
                feedback=(
                    f"`clickstream_replay` could not replay the trajectories: "
                    f"{type(exc).__name__}: {exc}. Skipping."
                ),
                details={
                    "task_id": ctx.selected_task_id,
                    "error": str(exc),
                    "skipped": True,
                },
            )

        # Step 5: aggregate + map the verdict.
        return _verdict_from_outcomes(
            outcomes=outcomes,
            sessions=len(sampled.sessions),
            dropped=sampled.dropped,
            prior_fails=prior_fails,
        )


# --------------------------------------------------------------------------- #
# Verdict aggregation
# --------------------------------------------------------------------------- #


def _verdict_from_outcomes(
    *,
    outcomes: Sequence[StepOutcome],
    sessions: int,
    dropped: int,
    prior_fails: int,
) -> VerifierResult:
    """Tally per-step outcomes into a :class:`VerifierResult`.

    Skipped steps are advisory: they neither count toward ``steps_total``
    nor gate the verdict. A FAIL is returned iff at least one non-skipped
    step was not reproducible.
    """
    checked = [o for o in outcomes if not o.skipped]
    doable = [o for o in checked if o.reproducible]
    failures = [o for o in checked if not o.reproducible]
    skipped_count = sum(1 for o in outcomes if o.skipped)

    per_session = _per_session_counts(outcomes)
    details: dict[str, Any] = {
        "ran": True,
        "sessions": sessions,
        "dropped": dropped,
        "steps_total": len(checked),
        "steps_skipped": skipped_count,
        "doable": len(doable),
        "not_doable": len(failures),
        "prior_fails": prior_fails,
        "per_session": per_session,
        "failures": [
            {
                "session_id": o.session_id,
                "step": o.step_index,
                "action": o.action,
                "url": o.url,
                "reason": o.reason,
            }
            for o in failures
        ],
    }

    if not failures:
        return VerifierResult(verdict=Verdict.PASS, details=details)
    return VerifierResult(
        verdict=Verdict.FAIL,
        feedback=_render_failure_markdown(failures=failures, checked=len(checked)),
        details=details,
    )


def _per_session_counts(outcomes: Sequence[StepOutcome]) -> list[dict[str, Any]]:
    """Per-session doable / not-doable / skipped tallies, ordered by id."""
    tallies: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        bucket = tallies.setdefault(
            outcome.session_id,
            {"doable": 0, "not_doable": 0, "skipped": 0},
        )
        if outcome.skipped:
            bucket["skipped"] += 1
        elif outcome.reproducible:
            bucket["doable"] += 1
        else:
            bucket["not_doable"] += 1
    return [{"session_id": session_id, **tallies[session_id]} for session_id in sorted(tallies)]


def _render_failure_markdown(*, failures: Sequence[StepOutcome], checked: int) -> str:
    """Render the FAIL feedback body surfaced at ``{{verifier_feedback}}``."""
    lines = [
        f"`clickstream_replay` failed: {len(failures)} of {checked} replayed "
        "step(s) could not be reproduced on the generated twin.",
        "",
    ]
    lines.extend(
        "- session `{session}` step {step} (`{action}`) → {url}: {reason}".format(
            session=o.session_id,
            step=o.step_index,
            action=o.action,
            url=f"`{o.url}`" if o.url else "(no url)",
            reason=o.reason,
        )
        for o in failures
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Production replayer (sync Playwright)
# --------------------------------------------------------------------------- #


class PlaywrightTrajectoryReplayer:
    """Default :class:`TrajectoryReplayer` driven by sync Playwright.

    Launches one headless Chromium and opens a fresh context per session
    (so carts stay isolated), then walks each session's events, mapping
    :class:`~shop_arena.gen.build.verifiers._clickstream.ClickstreamEvent`
    semantics onto concrete browser actions. Navigation / interaction
    errors on a single step are captured as a non-reproducible outcome
    rather than aborting the sweep.
    """

    def __call__(
        self,
        *,
        base_url: str,
        sessions: Sequence[Session],
        timeout_s: float,
    ) -> tuple[StepOutcome, ...]:
        """Replay every session in an isolated browser context."""
        timeout_ms = int(timeout_s * 1000)
        outcomes: list[StepOutcome] = []
        with playwright.sync_api.sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                for session in sessions:
                    context = browser.new_context()
                    try:
                        page = context.new_page()
                        outcomes.extend(
                            self._replay_session(
                                page=page,
                                base_url=base_url,
                                session=session,
                                timeout_ms=timeout_ms,
                            ),
                        )
                    finally:
                        context.close()
            finally:
                browser.close()
        return tuple(outcomes)

    def _replay_session(
        self,
        *,
        page: playwright.sync_api.Page,
        base_url: str,
        session: Session,
        timeout_ms: int,
    ) -> list[StepOutcome]:
        """Walk one session's events, returning one outcome per step."""
        results: list[StepOutcome] = []
        navigated = False
        for index, event in enumerate(session.events):
            outcome = self._replay_event(
                page=page,
                base_url=base_url,
                event=event,
                step_index=index,
                timeout_ms=timeout_ms,
                navigated=navigated,
            )
            if outcome.url is not None and outcome.reproducible:
                navigated = True
            results.append(outcome)
        return results

    def _replay_event(
        self,
        *,
        page: playwright.sync_api.Page,
        base_url: str,
        event: ClickstreamEvent,
        step_index: int,
        timeout_ms: int,
        navigated: bool,
    ) -> StepOutcome:
        """Reproduce a single event; capture any error as non-reproducible."""
        action = event.semantic_action
        url = event.target_url(base_url)
        try:
            match action:
                case s if s == ACTION_TERMINATE:
                    return _ok(event, step_index, url=None)
                case s if s == ACTION_EXPLORE_STAY:
                    reason = "" if navigated else "no page loaded to browse"
                    return _outcome(event, step_index, url=None, ok=navigated, reason=reason)
                case s if s == ACTION_DETAIL:
                    return self._replay_detail(page, event, step_index, url, timeout_ms)
                case s if s == ACTION_ADD:
                    return self._replay_add(page, base_url, event, step_index, url, timeout_ms)
                case _:
                    return self._replay_generic(page, event, step_index, url, timeout_ms)
        except playwright.sync_api.Error as exc:
            return _outcome(
                event,
                step_index,
                url=url,
                ok=False,
                reason=f"playwright error: {exc}",
            )

    def _replay_detail(
        self,
        page: playwright.sync_api.Page,
        event: ClickstreamEvent,
        step_index: int,
        url: str | None,
        timeout_ms: int,
    ) -> StepOutcome:
        """``detail`` — the PDP must respond ``< 400`` and render a product."""
        if url is None:
            return _skip(event, step_index, reason="detail step carried no url")
        response = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        status = response.status if response is not None else None
        if status is not None and status >= _HTTP_ERROR_FLOOR:
            return _outcome(event, step_index, url=url, ok=False, reason=f"HTTP {status}")
        if page.locator(_PRODUCT_HEADING_SELECTOR).count() == 0:
            return _outcome(
                event,
                step_index,
                url=url,
                ok=False,
                reason="product detail did not render (no heading)",
            )
        return _ok(event, step_index, url=url)

    def _replay_add(
        self,
        page: playwright.sync_api.Page,
        base_url: str,
        event: ClickstreamEvent,
        step_index: int,
        url: str | None,
        timeout_ms: int,
    ) -> StepOutcome:
        """``add`` — submit the add-to-cart form, then confirm the cart line."""
        if url is not None and page.url.rstrip("/") != url.rstrip("/"):
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        button = page.locator(_ADD_TO_CART_SELECTOR).first
        if button.count() == 0:
            return _outcome(
                event,
                step_index,
                url=url,
                ok=False,
                reason="no add-to-cart control found on the page",
            )
        # Templates gate the add button behind a required variant/size
        # selection (the react-vite VariantSelector disables the submit
        # until a size is chosen). Reproduce the real user flow — select
        # the first in-stock option — before submitting.
        if not button.is_enabled():
            self._select_variant(page, timeout_ms)
        if not button.is_enabled():
            return _outcome(
                event,
                step_index,
                url=url,
                ok=False,
                reason="add-to-cart control stayed disabled (no selectable variant)",
            )
        button.click(timeout=timeout_ms)
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        cart_response = page.goto(
            base_url.rstrip("/") + "/cart",
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        cart_status = cart_response.status if cart_response is not None else None
        if cart_status is not None and cart_status >= _HTTP_ERROR_FLOOR:
            return _outcome(event, step_index, url=url, ok=False, reason=f"cart HTTP {cart_status}")
        if not _cart_has_line(page, handle=event.product_handle):
            return _outcome(
                event,
                step_index,
                url=url,
                ok=False,
                reason="cart did not reflect the added line",
            )
        return _ok(event, step_index, url=url)

    def _select_variant(self, page: playwright.sync_api.Page, timeout_ms: int) -> None:
        """Best-effort: pick the first in-stock variant to enable add-to-cart.

        Opens the variant/size picker (if the template hides options behind
        a trigger) and clicks the first selectable option. Silently returns
        when no picker is present — the caller re-checks ``is_enabled`` and
        reports the disabled control as non-reproducible.
        """
        trigger = page.locator(_VARIANT_TRIGGER_SELECTOR).first
        if trigger.count() > 0:
            with contextlib.suppress(playwright.sync_api.Error):
                trigger.click(timeout=timeout_ms)
        option = page.locator(_VARIANT_OPTION_SELECTOR).first
        if option.count() > 0:
            with contextlib.suppress(playwright.sync_api.Error):
                option.click(timeout=timeout_ms)

    def _replay_generic(
        self,
        page: playwright.sync_api.Page,
        event: ClickstreamEvent,
        step_index: int,
        url: str | None,
        timeout_ms: int,
    ) -> StepOutcome:
        """Any other action: navigate to its path (if any); ``< 400`` = ok."""
        if url is None:
            return _skip(
                event,
                step_index,
                reason=f"action `{event.semantic_action}` carried no url",
            )
        response = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        status = response.status if response is not None else None
        if status is not None and status >= _HTTP_ERROR_FLOOR:
            return _outcome(event, step_index, url=url, ok=False, reason=f"HTTP {status}")
        return _ok(event, step_index, url=url)


def _cart_has_line(page: playwright.sync_api.Page, *, handle: str) -> bool:
    """Return ``True`` when the ``/cart`` page shows the added line.

    Prefers a handle-specific product link; falls back to *any* product
    link when the row carried no handle.
    """
    if handle:
        return page.locator(f"a[href*='/products/{handle}']").count() > 0
    return page.locator("a[href*='/products/']").count() > 0


# --------------------------------------------------------------------------- #
# Outcome constructors
# --------------------------------------------------------------------------- #


def _outcome(
    event: ClickstreamEvent,
    step_index: int,
    *,
    url: str | None,
    ok: bool,
    reason: str,
) -> StepOutcome:
    """Build a checked (non-skipped) :class:`StepOutcome`."""
    return StepOutcome(
        session_id=event.session_id,
        step_index=step_index,
        action=event.semantic_action,
        url=url,
        reproducible=ok,
        skipped=False,
        reason=reason,
    )


def _ok(event: ClickstreamEvent, step_index: int, *, url: str | None) -> StepOutcome:
    """Build a reproducible :class:`StepOutcome`."""
    return _outcome(event, step_index, url=url, ok=True, reason="")


def _skip(event: ClickstreamEvent, step_index: int, *, reason: str) -> StepOutcome:
    """Build a skipped (advisory) :class:`StepOutcome`."""
    return StepOutcome(
        session_id=event.session_id,
        step_index=step_index,
        action=event.semantic_action,
        url=None,
        reproducible=True,
        skipped=True,
        reason=reason,
    )


__all__ = [
    "ClickstreamReplayVerifier",
    "PlaywrightTrajectoryReplayer",
    "StepOutcome",
    "TrajectoryReplayer",
]
