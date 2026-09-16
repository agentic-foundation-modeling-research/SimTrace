"""``visual_fidelity`` build-loop verifier.

Side-by-side visual-fidelity gate. After the executor generates (or
fixes) a page, this verifier compares the *generated* storefront's
freshly rendered screenshots against *reference* screenshots captured
from the original source storefront during exploration, asks a vision
LLM to score structural + design fidelity, and — on a low score —
feeds actionable fix instructions back into the next executor
iteration via the harness ``{{verifier_feedback}}`` slot.

Unlike :class:`~shop_arena.gen.build.verifiers.visual_judge.VisualJudgeVerifier`
(which judges the generated page against the shop's *capabilities*
document), this verifier grounds its judgement in the *original*
storefront's look-and-feel. The reference images are the best-effort
agent screenshots the exploration phase left under
``<seed>/artifact/evidence/<evidence_dir>/screenshots/*.png``. That tree
is read **read-only** — it is neither part of the published gen/explore
contract nor copied into the build workspace, so the verifier is handed
the seed directories at construction time (the
:class:`~harness.verifiers.VerifierContext` does not carry them).

The verifier reads:

* ``<seed>/artifact/evidence/<evidence_dir>/screenshots/*.png`` — the
  reference screenshots for the active task's page bucket(s), across
  every configured seed directory.
* ``data_dir / {collections,products,pages}.json`` — supplied at
  construction time, used by
  :func:`shop_arena.gen.build.verifiers._task_routes.routes_for_buckets`
  to materialise the route list the generated storefront is captured at.
* ``ctx.artifact_dir / "hydrogen"`` — the dev-server root the injected
  :class:`~shop_arena.gen.final_eval.playwright_smoke.DevServerFactory`
  boots against.

Failure handling never crashes the loop. A missing hydrogen tree or an
unresolvable task surfaces as ``FAIL`` / ``ERROR``; missing reference
evidence (the exploration screenshots are best-effort), missing API
credentials (:class:`~shop_arena.util._llm.LLMConfigError`), and a
capture that produced nothing all surface as ``ADVISORY`` so the loop
proceeds rather than wedging on an environment gap.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import playwright.sync_api
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.verifiers import Verdict, VerifierContext, VerifierResult
from shop_arena.gen.build.prompts import load_visual_fidelity_judge_prompt
from shop_arena.gen.build.verifiers._history import count_prior_task_fails
from shop_arena.gen.build.verifiers._task_routes import (
    DEFAULT_CAPS,
    buckets_for_task,
    routes_for_buckets,
)
from shop_arena.gen.final_eval.playwright_smoke import DevServerFactory
from shop_arena.util._llm import (
    LLMConfigError,
    LLMVisionClient,
    VisionResponse,
    build_default_client,
)

_log = logging.getLogger(__name__)

_NAME: Final[str] = "visual_fidelity"
"""Verifier name (filesystem-safe; matches the spec table)."""

_DEFAULT_FIDELITY_MODEL: Final[str] = "claude-sonnet-4-6"
"""Default vision model id (mirrors ``env_eval``'s ``DEFAULT_RUBRIC_MODEL``)."""

_DEFAULT_RETRY_BUDGET: Final[int] = 3
"""Per-task cap on consecutive ``visual_fidelity`` FAILs (mirrors ``visual_judge``)."""

_DEFAULT_PASS_THRESHOLD: Final[float] = 7.0
"""Overall-fidelity score floor below which the verdict is FAIL."""

_DEFAULT_TIMEOUT_S: Final[float] = 180.0
"""Wall-clock budget for a single route's Playwright navigation + capture."""

_DEFAULT_VIEWPORT: Final[tuple[int, int]] = (1440, 900)
"""Desktop capture viewport (mirrors ``env_eval``'s ``DEFAULT_VIEWPORT``)."""

_HYDROGEN_DIRNAME: Final[str] = "hydrogen"
"""Subdir of ``ctx.artifact_dir`` the dev-server factory is rooted at."""

_EVIDENCE_PARTS: Final[tuple[str, ...]] = ("artifact", "evidence")
"""Sub-path inside a seed dir where exploration wrote agent screenshots."""

_SCREENSHOTS_DIRNAME: Final[str] = "screenshots"
"""Leaf dir holding the PNGs under each evidence task dir."""

_MAX_REFERENCE_IMAGES: Final[int] = 4
"""Cap on reference PNGs forwarded to the model (bounds prompt token cost)."""

_MAX_GENERATED_IMAGES: Final[int] = 4
"""Cap on freshly captured generated PNGs forwarded to the model."""

#: Page-bucket → exploration-evidence task directory names. The union over
#: the active task's buckets (resolved via :func:`buckets_for_task`, which
#: already strips a ``_redo_<n>`` suffix) yields the reference dirs to glob.
#: Keys mirror :data:`shop_arena.gen.build.verifiers._task_routes.TASK_BUCKETS`
#: values; the directory names match what
#: :mod:`shop_arena.explore` writes on disk.
_BUCKET_EVIDENCE_DIRS: Final[dict[str, tuple[str, ...]]] = {
    "homepage": ("homepage_sections",),
    "navigation": ("header_navigation",),
    "collections": ("collection_filters",),
    "product": ("product_variants",),
    "cart_search": ("cart_drawer", "search_predictive"),
    "info_pages": ("info_pages",),
}

_DEFAULT_TASKS: Final[frozenset[str]] = frozenset(
    {
        "gen_homepage",
        "gen_navigation",
        "gen_collections",
        "gen_product",
        "gen_cart_search",
        "gen_info_pages",
        "visual_fix",
    },
)
"""Tasks this verifier applies to (the six page-gen tasks plus ``visual_fix``)."""


class _CategoryScores(BaseModel):
    """Per-dimension 0-10 fidelity sub-scores emitted by the model."""

    model_config = ConfigDict(extra="forbid")

    layout: float = Field(ge=0.0, le=10.0)
    color_typography: float = Field(ge=0.0, le=10.0)
    components: float = Field(ge=0.0, le=10.0)
    content_density: float = Field(ge=0.0, le=10.0)


class _LanguageMatch(BaseModel):
    """Detected primary language of each set and whether they agree."""

    model_config = ConfigDict(extra="forbid")

    reference_language: str
    generated_language: str
    match: bool


class _FidelityVerdict(BaseModel):
    """Closed schema the vision model must emit (validated post-call)."""

    model_config = ConfigDict(extra="forbid")

    overall_fidelity_score: float = Field(ge=0.0, le=10.0)
    category_scores: _CategoryScores
    language: _LanguageMatch
    critical_issues: tuple[str, ...]
    major_issues: tuple[str, ...]
    minor_issues: tuple[str, ...]
    fix_instructions: tuple[str, ...]
    summary: str


#: JSON schema forwarded to the vision client. Kept in lockstep with
#: :class:`_FidelityVerdict`; every key is ``required`` and
#: ``additionalProperties`` is ``false`` so the payload works under
#: OpenAI's strict ``json_schema`` mode (and Anthropic tool-use).
_FIDELITY_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "overall_fidelity_score": {"type": "number", "minimum": 0, "maximum": 10},
        "category_scores": {
            "type": "object",
            "properties": {
                "layout": {"type": "number", "minimum": 0, "maximum": 10},
                "color_typography": {"type": "number", "minimum": 0, "maximum": 10},
                "components": {"type": "number", "minimum": 0, "maximum": 10},
                "content_density": {"type": "number", "minimum": 0, "maximum": 10},
            },
            "required": ["layout", "color_typography", "components", "content_density"],
            "additionalProperties": False,
        },
        "language": {
            "type": "object",
            "properties": {
                "reference_language": {"type": "string"},
                "generated_language": {"type": "string"},
                "match": {"type": "boolean"},
            },
            "required": ["reference_language", "generated_language", "match"],
            "additionalProperties": False,
        },
        "critical_issues": {"type": "array", "items": {"type": "string"}},
        "major_issues": {"type": "array", "items": {"type": "string"}},
        "minor_issues": {"type": "array", "items": {"type": "string"}},
        "fix_instructions": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": [
        "overall_fidelity_score",
        "category_scores",
        "language",
        "critical_issues",
        "major_issues",
        "minor_issues",
        "fix_instructions",
        "summary",
    ],
    "additionalProperties": False,
}


@runtime_checkable
class ScreenshotCapturer(Protocol):
    """Captures full-page screenshots of ``routes`` served at ``base_url``.

    Production wiring is :func:`_capture_screenshots` (sync Playwright);
    tests inject a stub that fabricates deterministic PNG bytes so the
    verifier can be exercised without launching Chromium.
    """

    def __call__(
        self,
        *,
        base_url: str,
        routes: Sequence[str],
        out_dir: Path,
        viewport: tuple[int, int],
        timeout_s: float,
    ) -> tuple[Path, ...]:
        """Navigate each route, write one PNG per route into ``out_dir``.

        Returns the paths of the screenshots that were captured
        successfully (routes that failed to load are skipped, not
        raised).
        """
        ...


class VisualFidelityVerifier:
    """Compares generated pages against source-storefront reference shots.

    Attributes:
        name: ``"visual_fidelity"`` — used as the per-verifier telemetry
            filename and the markdown section heading in ``feedback.md``.
    """

    name: str = _NAME

    def __init__(
        self,
        *,
        data_dir: Path,
        seed_dirs: tuple[Path, ...],
        dev_server_factory: DevServerFactory,
        vision_client_factory: Callable[[], LLMVisionClient] = lambda: build_default_client(
            _DEFAULT_FIDELITY_MODEL,
        ),
        capture: ScreenshotCapturer | None = None,
        app_dir: Path = Path(_HYDROGEN_DIRNAME),
        retry_budget: int = _DEFAULT_RETRY_BUDGET,
        pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        viewport: tuple[int, int] = _DEFAULT_VIEWPORT,
        applicable_tasks: Iterable[str] | None = None,
    ) -> None:
        """Build the verifier with optional injection seams.

        Args:
            data_dir: Directory containing the published
                ``collections.json`` / ``products.json`` / ``pages.json``
                files. Forwarded to
                :func:`shop_arena.gen.build.verifiers._task_routes.routes_for_buckets`.
            seed_dirs: Seed (shop-manual) directories whose
                ``artifact/evidence/`` trees hold the reference
                screenshots. Read-only.
            dev_server_factory: Shared dev-server factory; production
                wiring is the ``pnpm dev`` runner, tests inject a stub
                yielding a deterministic base URL.
            vision_client_factory: Zero-arg factory returning the vision
                client used for the comparison. Defaults to
                :func:`shop_arena.util._llm.build_default_client` against
                :data:`_DEFAULT_FIDELITY_MODEL` (credentials read from
                ``ANTHROPIC_API_KEY`` at call time).
            capture: Screenshot capturer for the generated pages.
                Defaults to :func:`_capture_screenshots` (sync
                Playwright).
            app_dir: Storefront app directory relative to
                ``VerifierContext.artifact_dir``. Defaults to ``hydrogen``.
            retry_budget: Per-task cap on consecutive FAILs before the
                verifier downgrades to ADVISORY. ``0`` disables the
                budget.
            pass_threshold: Overall-fidelity score floor for PASS.
            timeout_s: Per-route Playwright navigation budget (seconds).
            viewport: Desktop capture viewport ``(width, height)``.
            applicable_tasks: Override the default applicability set.
                Defaults to :data:`_DEFAULT_TASKS`.
        """
        self._data_dir = data_dir
        self._seed_dirs = seed_dirs
        self._dev_server_factory = dev_server_factory
        self._vision_client_factory = vision_client_factory
        self._capture: ScreenshotCapturer = capture if capture is not None else _capture_screenshots
        self._app_dir = app_dir
        self._retry_budget = retry_budget
        self._pass_threshold = pass_threshold
        self._timeout_s = timeout_s
        self._viewport = viewport
        self._applicable_tasks: frozenset[str] = (
            _DEFAULT_TASKS if applicable_tasks is None else frozenset(applicable_tasks)
        )

    def applies_to(self, task_id: str) -> bool:
        """Match the configured applicability set.

        Args:
            task_id: Selected task id.

        Returns:
            ``True`` when this verifier should run for ``task_id``.
        """
        return task_id in self._applicable_tasks

    def run(  # noqa: PLR0911 -- one early-return per precondition
        self,
        ctx: VerifierContext,
    ) -> VerifierResult:
        """Capture the generated pages, LLM-compare to references, map the verdict.

        Args:
            ctx: Verifier context. Reads ``ctx.artifact_dir`` for the
                hydrogen tree; ``ctx.run_dir`` / ``ctx.iter_id`` for the
                retry-budget scan and telemetry output.

        Returns:
            ``PASS`` when the overall fidelity score meets the threshold,
            no critical issue is reported, and the generated storefront's
            language matches the reference's; ``FAIL`` (with actionable
            feedback) otherwise. Missing reference evidence, missing API
            credentials, an exhausted retry budget, and a capture that
            produced nothing all surface as ``ADVISORY``. An unknown task
            or an unparseable model response surfaces as ``ERROR``.
        """
        # Step 1: resolve task scope.
        buckets = buckets_for_task(ctx.selected_task_id)
        if not buckets:
            return VerifierResult(
                verdict=Verdict.ERROR,
                feedback=(
                    f"`visual_fidelity` does not know how to scope task "
                    f"`{ctx.selected_task_id}`; no page-bucket mapping is registered."
                ),
                details={"task_id": ctx.selected_task_id, "buckets_run": []},
            )
        buckets_run = sorted(buckets)

        hydrogen_dir = ctx.artifact_dir / self._app_dir
        if not hydrogen_dir.is_dir():
            return VerifierResult(
                verdict=Verdict.FAIL,
                feedback=(
                    f"`visual_fidelity` could not find the storefront tree at "
                    f"`{self._app_dir.as_posix()}/`. Did `clone_template` run?"
                ),
                details={"hydrogen_dir": str(hydrogen_dir), "exists": False},
            )

        routes = routes_for_buckets(buckets, self._data_dir, caps=DEFAULT_CAPS)
        if not routes:
            return VerifierResult(
                verdict=Verdict.FAIL,
                feedback=(
                    f"`visual_fidelity` resolved no routes for task "
                    f"`{ctx.selected_task_id}` (buckets={buckets_run}); the data "
                    "directory may be empty or the bucket map is out of sync with "
                    "the dataset."
                ),
                details={
                    "task_id": ctx.selected_task_id,
                    "buckets_run": buckets_run,
                    "data_dir": str(self._data_dir),
                },
            )

        # Step 2: load reference screenshots. Exploration evidence is
        # best-effort — when a task's dirs are missing or empty there is
        # nothing to compare against, so ADVISE-skip rather than FAIL.
        reference_paths = self._collect_reference_paths(buckets)
        if not reference_paths:
            return VerifierResult(
                verdict=Verdict.ADVISORY,
                feedback=(
                    f"`visual_fidelity` found no reference screenshots for task "
                    f"`{ctx.selected_task_id}` (buckets={buckets_run}) under the "
                    "seed evidence trees; skipping the fidelity comparison."
                ),
                details={
                    "task_id": ctx.selected_task_id,
                    "buckets_run": buckets_run,
                    "reference_count": 0,
                    "skipped": True,
                },
            )

        # Step 3: per-task retry budget. Downgrade to ADVISORY to break
        # the loop before spending an LLM call or booting the dev server.
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
                    f"`visual_fidelity` has FAILed {prior_fails} time(s) against "
                    f"task `{ctx.selected_task_id}`, meeting the configured retry "
                    f"budget ({self._retry_budget}). Downgrading to ADVISORY to "
                    "break the loop."
                ),
                details={
                    "task_id": ctx.selected_task_id,
                    "buckets_run": buckets_run,
                    "retry_budget": self._retry_budget,
                    "retry_budget_exhausted": True,
                    "prior_fails": prior_fails,
                },
            )

        parent_dir = ctx.run_dir / "iters" / ctx.iter_id / "checks" / "verifiers" / self.name
        parent_dir.mkdir(parents=True, exist_ok=True)

        common_details: dict[str, Any] = {
            "task_id": ctx.selected_task_id,
            "buckets_run": buckets_run,
            "routes": list(routes),
            "reference_count": len(reference_paths),
            "retry_budget_exhausted": False,
            "prior_fails": prior_fails,
        }

        # Step 4: boot the dev server + capture the generated pages.
        generated_dir = parent_dir / _SCREENSHOTS_DIRNAME
        _log.info(
            "visual_fidelity starting task=%s buckets=%s routes=%d refs=%d",
            ctx.selected_task_id,
            buckets_run,
            len(routes),
            len(reference_paths),
        )
        with self._dev_server_factory(hydrogen_dir) as base_url:
            generated_paths = self._capture(
                base_url=base_url,
                routes=routes,
                out_dir=generated_dir,
                viewport=self._viewport,
                timeout_s=self._timeout_s,
            )
        if not generated_paths:
            return VerifierResult(
                verdict=Verdict.ADVISORY,
                feedback=(
                    f"`visual_fidelity` captured no generated screenshots for task "
                    f"`{ctx.selected_task_id}` (routes={list(routes)}); the dev "
                    "server may be unreachable. `routes_200` is the cheap gate for "
                    "this. Skipping the fidelity comparison."
                ),
                details={**common_details, "generated_count": 0, "skipped": True},
            )

        reference_bytes = _read_pngs(reference_paths[:_MAX_REFERENCE_IMAGES])
        generated_bytes = _read_pngs(generated_paths[:_MAX_GENERATED_IMAGES])
        common_details["generated_count"] = len(generated_bytes)
        common_details["reference_count_sent"] = len(reference_bytes)

        # Step 5: vision comparison. Missing credentials must not crash
        # the loop — degrade to ADVISORY.
        prompt = load_visual_fidelity_judge_prompt().format(
            task_id=ctx.selected_task_id,
            buckets=", ".join(buckets_run),
            route_list=_render_route_list(routes),
            reference_count=len(reference_bytes),
            generated_count=len(generated_bytes),
            verdict_schema=json.dumps(_FIDELITY_SCHEMA, indent=2),
        )
        try:
            client = self._vision_client_factory()
            response = client.call(
                prompt=prompt,
                images=(*reference_bytes, *generated_bytes),
                schema=_FIDELITY_SCHEMA,
            )
        except LLMConfigError as exc:
            _log.warning("visual_fidelity: vision client unavailable: %s", exc)
            return VerifierResult(
                verdict=Verdict.ADVISORY,
                feedback=(
                    "`visual_fidelity` could not run the vision comparison: "
                    f"{exc}. Set `ANTHROPIC_API_KEY` to enable it. Skipping."
                ),
                details={**common_details, "llm_config_error": str(exc), "skipped": True},
            )

        verdict = _parse_verdict(response)
        if verdict is None:
            return VerifierResult(
                verdict=Verdict.ERROR,
                feedback=(
                    "`visual_fidelity` could not parse the vision model's response "
                    f"against the expected schema. Parse errors: "
                    f"{'; '.join(response.parse_errors) or 'schema validation failed'}."
                ),
                details={
                    **common_details,
                    "parse_errors": list(response.parse_errors),
                    "phase": "parse",
                },
            )

        _write_verdict(parent_dir / "verdict.json", verdict)

        passed = (
            verdict.overall_fidelity_score >= self._pass_threshold
            and not verdict.critical_issues
            and verdict.language.match
        )
        return VerifierResult(
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            feedback="" if passed else _compose_feedback(verdict, self._pass_threshold),
            details={
                **common_details,
                "overall_fidelity_score": verdict.overall_fidelity_score,
                "category_scores": verdict.category_scores.model_dump(),
                "language": verdict.language.model_dump(),
                "critical_issue_count": len(verdict.critical_issues),
                "major_issue_count": len(verdict.major_issues),
                "minor_issue_count": len(verdict.minor_issues),
            },
        )

    def _collect_reference_paths(self, buckets: Iterable[str]) -> tuple[Path, ...]:
        """Glob reference PNGs for ``buckets`` across every seed dir.

        Walks the union of :data:`_BUCKET_EVIDENCE_DIRS` for the active
        buckets, under each seed's ``artifact/evidence/<dir>/screenshots/``.
        Results are sorted for determinism.
        """
        evidence_dirs: set[str] = set()
        for bucket in buckets:
            evidence_dirs.update(_BUCKET_EVIDENCE_DIRS.get(bucket, ()))
        paths: list[Path] = []
        for seed_dir in self._seed_dirs:
            evidence_root = seed_dir.joinpath(*_EVIDENCE_PARTS)
            for name in evidence_dirs:
                shots_dir = evidence_root / name / _SCREENSHOTS_DIRNAME
                if not shots_dir.is_dir():
                    continue
                paths.extend(p for p in shots_dir.glob("*.png") if p.is_file())
        return tuple(sorted(paths))


def _capture_screenshots(
    *,
    base_url: str,
    routes: Sequence[str],
    out_dir: Path,
    viewport: tuple[int, int],
    timeout_s: float,
) -> tuple[Path, ...]:
    """Full-page-capture every route with sync Playwright (production wiring).

    Launches a headless Chromium, opens one context at ``viewport``, and
    navigates to each ``base_url + route`` in turn. A route that fails to
    load (timeout, navigation error) is logged and skipped rather than
    aborting the sweep, so a single broken page never denies the model
    the pages that did render. The browser is torn down on every exit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout_ms = int(timeout_s * 1000)
    captured: list[Path] = []
    with playwright.sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
            )
            page = context.new_page()
            for route in routes:
                url = base_url.rstrip("/") + route
                try:
                    page.goto(url, timeout=timeout_ms, wait_until="networkidle")
                except playwright.sync_api.Error as exc:
                    _log.warning("visual_fidelity: capture of %s failed: %s", url, exc)
                    continue
                dest = out_dir / f"{_route_slug(route)}.png"
                page.screenshot(path=str(dest), full_page=True)
                captured.append(dest)
        finally:
            browser.close()
    return tuple(captured)


def _route_slug(route: str) -> str:
    """Turn a route path into a filesystem-safe screenshot stem."""
    slug = route.strip("/").replace("/", "-").replace("?", "__").replace("=", "-").replace("&", "-")
    return slug or "home"


def _render_route_list(routes: Iterable[str]) -> str:
    """Render the resolved routes as a markdown bullet list."""
    return "\n".join(f"- `{route}`" for route in routes)


def _read_pngs(paths: Sequence[Path]) -> tuple[bytes, ...]:
    """Read each PNG path into bytes, skipping any that fail to read."""
    out: list[bytes] = []
    for path in paths:
        try:
            out.append(path.read_bytes())
        except OSError as exc:  # pragma: no cover -- defensive
            _log.warning("visual_fidelity: could not read %s: %s", path, exc)
    return tuple(out)


def _parse_verdict(response: VisionResponse) -> _FidelityVerdict | None:
    """Validate ``response.parsed`` against :class:`_FidelityVerdict`.

    Returns ``None`` when the client failed to decode a JSON object or
    when the closed pydantic schema rejects it.
    """
    if response.parsed is None:
        return None
    try:
        return _FidelityVerdict.model_validate(dict(response.parsed))
    except ValidationError as exc:
        _log.warning("visual_fidelity: verdict failed schema validation: %s", exc)
        return None


def _write_verdict(path: Path, verdict: _FidelityVerdict) -> None:
    """Persist the parsed verdict next to the screenshots for audit."""
    path.write_text(
        json.dumps(verdict.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _compose_feedback(verdict: _FidelityVerdict, pass_threshold: float) -> str:
    """Build the markdown feedback body surfaced to the next iteration.

    Leads with the score + summary, flags a language mismatch when the
    generated copy is not in the reference's language, then enumerates
    issues (critical → minor) and the model's concrete fix instructions
    — the actionable half the executor consumes at
    ``{{verifier_feedback}}``.
    """
    lines: list[str] = [
        f"visual fidelity score: {verdict.overall_fidelity_score:.1f} "
        f"(threshold: {pass_threshold:.1f})",
    ]
    if verdict.summary:
        lines.append("")
        lines.append(verdict.summary)
    if not verdict.language.match:
        lines.append("")
        lines.append("## Language mismatch")
        lines.append(
            f"- generated storefront is in {verdict.language.generated_language} but the "
            f"reference is in {verdict.language.reference_language}; rewrite all "
            f"user-facing copy in {verdict.language.reference_language}."
        )
    for label, issues in (
        ("Critical", verdict.critical_issues),
        ("Major", verdict.major_issues),
        ("Minor", verdict.minor_issues),
    ):
        if issues:
            lines.append("")
            lines.append(f"## {label} issues")
            lines.extend(f"- {issue}" for issue in issues)
    if verdict.fix_instructions:
        lines.append("")
        lines.append("## Fix instructions")
        lines.extend(f"- {fix}" for fix in verdict.fix_instructions)
    return "\n".join(lines)


__all__ = ["VisualFidelityVerifier"]
