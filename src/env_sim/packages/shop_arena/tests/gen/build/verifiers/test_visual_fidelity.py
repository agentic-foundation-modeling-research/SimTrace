"""Unit tests for :class:`shop_arena.gen.build.verifiers.visual_fidelity.VisualFidelityVerifier`.

Exercise the verifier against injected seams so no browser is launched
and no model is queried:

* a recording :class:`~shop_arena.gen.final_eval.playwright_smoke.DevServerFactory`
  stub (lifecycle counters),
* a stub ``ScreenshotCapturer`` that fabricates deterministic PNG files,
* a stub :class:`~shop_arena.util._llm.LLMVisionClient` returning canned
  :class:`~shop_arena.util._llm.VisionResponse` bodies.

Reference evidence is materialised on disk under a temp seed directory
so the read-only seed glob has real files to find.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from harness.verifiers import Verdict, VerifierContext
from shop_arena.gen.build.verifiers.visual_fidelity import VisualFidelityVerifier
from shop_arena.util._llm import LLMConfigError, VisionResponse

# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


@dataclass
class _StubDevServer:
    """Recording :class:`DevServerFactory` stub; tracks balanced lifecycle."""

    base_url: str = "http://127.0.0.1:8000"
    enters: int = 0
    exits: int = 0

    def __call__(self, hydrogen_dir: Path) -> _DevServerCtx:
        del hydrogen_dir
        return _DevServerCtx(self)


class _DevServerCtx:
    """Context manager incrementing the stub's lifecycle counters."""

    def __init__(self, server: _StubDevServer) -> None:
        self._server = server

    def __enter__(self) -> str:
        self._server.enters += 1
        return self._server.base_url

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._server.exits += 1


@dataclass
class _StubCapture:
    """``ScreenshotCapturer`` stub that writes one PNG per route."""

    n_shots: int = 2
    routes_seen: tuple[str, ...] = ()

    def __call__(
        self,
        *,
        base_url: str,
        routes: Sequence[str],
        out_dir: Path,
        viewport: tuple[int, int],
        timeout_s: float,
    ) -> tuple[Path, ...]:
        del base_url, viewport, timeout_s
        self.routes_seen = tuple(routes)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i in range(self.n_shots):
            dest = out_dir / f"shot-{i}.png"
            dest.write_bytes(f"gen-png-{i}".encode())
            paths.append(dest)
        return tuple(paths)


@dataclass
class _StubVisionClient:
    """``LLMVisionClient`` stub returning a canned response, recording calls."""

    response: VisionResponse
    images_seen: int = 0

    @property
    def model(self) -> str:
        return "claude-sonnet-4-6"

    def call(
        self,
        *,
        prompt: str,
        images: Sequence[bytes],
        schema: Mapping[str, Any],
        temperature: float = 0.0,
    ) -> VisionResponse:
        del prompt, schema, temperature
        self.images_seen = len(images)
        return self.response

    def call_text(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        temperature: float = 0.0,
    ) -> VisionResponse:
        del prompt, schema, temperature
        raise AssertionError("visual_fidelity never issues a text-only call")


# --------------------------------------------------------------------------- #
# Helpers + fixtures
# --------------------------------------------------------------------------- #


def _verdict_payload(
    *,
    score: float,
    critical: tuple[str, ...] = (),
    major: tuple[str, ...] = (),
    minor: tuple[str, ...] = (),
    fixes: tuple[str, ...] = (),
    summary: str = "looks close",
    reference_language: str = "English",
    generated_language: str = "English",
    language_match: bool = True,
) -> dict[str, Any]:
    """Build a schema-valid fidelity verdict body."""
    return {
        "overall_fidelity_score": score,
        "category_scores": {
            "layout": score,
            "color_typography": score,
            "components": score,
            "content_density": score,
        },
        "language": {
            "reference_language": reference_language,
            "generated_language": generated_language,
            "match": language_match,
        },
        "critical_issues": list(critical),
        "major_issues": list(major),
        "minor_issues": list(minor),
        "fix_instructions": list(fixes),
        "summary": summary,
    }


def _response(
    payload: Mapping[str, Any] | None,
    *,
    parse_errors: tuple[str, ...] = (),
) -> VisionResponse:
    return VisionResponse(
        parsed=payload,
        raw_response=json.dumps(payload) if payload is not None else "",
        parse_errors=parse_errors,
    )


def _client_factory(client: _StubVisionClient) -> Callable[[], _StubVisionClient]:
    return lambda: client


def _seed_reference(seed_dir: Path, evidence_name: str, count: int = 2) -> None:
    """Materialise ``count`` reference PNGs under a seed's evidence tree."""
    shots = seed_dir / "artifact" / "evidence" / evidence_name / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (shots / f"{i:02d}-state.png").write_bytes(f"ref-{evidence_name}-{i}".encode())


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Empty data dir; the homepage bucket resolves ``/`` without data files."""
    target = tmp_path / "data"
    target.mkdir()
    return target


@pytest.fixture
def seed_dir(tmp_path: Path) -> Path:
    """A seed (shop-manual) directory the verifier reads evidence from."""
    target = tmp_path / "seed"
    target.mkdir()
    return target


@pytest.fixture
def make_fidelity_ctx(
    make_ctx: Callable[..., VerifierContext],
) -> Callable[..., VerifierContext]:
    """Wrapper that pre-creates the ``iters/<iter_id>/`` telemetry dir."""

    def _factory(**kwargs: object) -> VerifierContext:
        ctx = make_ctx(**kwargs)
        (ctx.run_dir / "iters" / ctx.iter_id).mkdir(parents=True, exist_ok=True)
        return ctx

    return _factory


def _build_verifier(
    *,
    data_dir: Path,
    seed_dir: Path,
    client: _StubVisionClient | None,
    capture: _StubCapture | None = None,
    dev_server: _StubDevServer | None = None,
    client_factory: Callable[[], Any] | None = None,
    retry_budget: int = 3,
    pass_threshold: float = 7.0,
) -> VisualFidelityVerifier:
    if client_factory is None:
        assert client is not None
        client_factory = _client_factory(client)
    return VisualFidelityVerifier(
        data_dir=data_dir,
        seed_dirs=(seed_dir,),
        dev_server_factory=dev_server or _StubDevServer(),
        vision_client_factory=client_factory,
        capture=capture or _StubCapture(),
        retry_budget=retry_budget,
        pass_threshold=pass_threshold,
    )


# --------------------------------------------------------------------------- #
# Identity / applicability
# --------------------------------------------------------------------------- #


def test_name_and_applicability(data_dir: Path, seed_dir: Path) -> None:
    verifier = _build_verifier(
        data_dir=data_dir, seed_dir=seed_dir, client=None, client_factory=lambda: None
    )
    assert verifier.name == "visual_fidelity"
    for task_id in (
        "gen_homepage",
        "gen_navigation",
        "gen_collections",
        "gen_product",
        "gen_cart_search",
        "gen_info_pages",
        "visual_fix",
    ):
        assert verifier.applies_to(task_id) is True, f"missing task {task_id}"
    assert verifier.applies_to("gen_theme") is False
    assert verifier.applies_to("plan") is False


# --------------------------------------------------------------------------- #
# Verdict mapping
# --------------------------------------------------------------------------- #


def test_pass_on_high_score(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections", count=3)
    client = _StubVisionClient(response=_response(_verdict_payload(score=8.5)))
    capture = _StubCapture(n_shots=2)
    dev = _StubDevServer()
    verifier = _build_verifier(
        data_dir=data_dir, seed_dir=seed_dir, client=client, capture=capture, dev_server=dev
    )

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.PASS
    assert result.feedback == ""
    assert result.details["overall_fidelity_score"] == 8.5
    # References lead, generated follow; both were forwarded to the model.
    assert client.images_seen == 3 + 2
    assert result.details["generated_count"] == 2
    # Dev server lifecycle balanced.
    assert dev.enters == 1
    assert dev.exits == 1
    # Telemetry written.
    ctx = make_fidelity_ctx(selected_task_id="gen_homepage")
    verdict_path = (
        ctx.run_dir
        / "iters"
        / ctx.iter_id
        / "checks"
        / "verifiers"
        / "visual_fidelity"
        / "verdict.json"
    )
    assert verdict_path.is_file()


def test_fail_on_low_score_carries_fixes(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    client = _StubVisionClient(
        response=_response(
            _verdict_payload(
                score=3.0,
                major=("hero section is missing",),
                fixes=("add a full-bleed hero above the grid",),
            ),
        ),
    )
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.FAIL
    assert "hero section is missing" in result.feedback
    assert "add a full-bleed hero above the grid" in result.feedback
    assert "3.0" in result.feedback


def test_fail_on_critical_issue_even_above_threshold(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    client = _StubVisionClient(
        response=_response(
            _verdict_payload(score=9.0, critical=("entire product grid absent",)),
        ),
    )
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.FAIL
    assert "entire product grid absent" in result.feedback


def test_fail_on_language_mismatch_even_above_threshold(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    client = _StubVisionClient(
        response=_response(
            _verdict_payload(
                score=9.0,
                reference_language="French",
                generated_language="English",
                language_match=False,
            ),
        ),
    )
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.FAIL
    assert "Language mismatch" in result.feedback
    assert "French" in result.feedback
    assert result.details["language"] == {
        "reference_language": "French",
        "generated_language": "English",
        "match": False,
    }


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #


def test_advisory_when_no_reference_evidence(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    # No _seed_reference call → empty evidence tree.
    client = _StubVisionClient(response=_response(_verdict_payload(score=9.0)))
    capture = _StubCapture()
    dev = _StubDevServer()
    verifier = _build_verifier(
        data_dir=data_dir, seed_dir=seed_dir, client=client, capture=capture, dev_server=dev
    )

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.ADVISORY
    assert result.details["skipped"] is True
    # Short-circuited before booting the dev server / capturing / calling the model.
    assert dev.enters == 0
    assert capture.routes_seen == ()
    assert client.images_seen == 0


def test_advisory_on_missing_credentials(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")

    def _raising_factory() -> Any:
        raise LLMConfigError("ANTHROPIC_API_KEY is not set")

    dev = _StubDevServer()
    verifier = _build_verifier(
        data_dir=data_dir,
        seed_dir=seed_dir,
        client=None,
        dev_server=dev,
        client_factory=_raising_factory,
    )

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.ADVISORY
    assert result.details["skipped"] is True
    assert "ANTHROPIC_API_KEY" in result.feedback
    # Dev server still torn down cleanly.
    assert dev.enters == 1
    assert dev.exits == 1


def test_advisory_when_capture_empty(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    client = _StubVisionClient(response=_response(_verdict_payload(score=9.0)))
    capture = _StubCapture(n_shots=0)
    verifier = _build_verifier(
        data_dir=data_dir, seed_dir=seed_dir, client=client, capture=capture
    )

    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))

    assert result.verdict is Verdict.ADVISORY
    assert result.details["generated_count"] == 0
    assert client.images_seen == 0  # never reached the model


def test_retry_budget_downgrades_to_advisory(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    client = _StubVisionClient(response=_response(_verdict_payload(score=1.0)))
    capture = _StubCapture()
    ctx = make_fidelity_ctx(selected_task_id="gen_homepage", iter_id="exec-0004")
    # Seed three prior FAIL records for this (verifier, task) pair.
    for i in range(3):
        rec_dir = ctx.run_dir / "iters" / f"exec-000{i}" / "checks" / "verifiers"
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "visual_fidelity.json").write_text(
            json.dumps({"verdict": "fail", "task_id": "gen_homepage"}),
            encoding="utf-8",
        )
    verifier = _build_verifier(
        data_dir=data_dir, seed_dir=seed_dir, client=client, capture=capture, retry_budget=3
    )

    result = verifier.run(ctx)

    assert result.verdict is Verdict.ADVISORY
    assert result.details["retry_budget_exhausted"] is True
    assert result.details["prior_fails"] == 3
    assert client.images_seen == 0  # short-circuited before the model call


# --------------------------------------------------------------------------- #
# Hard failures
# --------------------------------------------------------------------------- #


def test_error_on_unknown_task(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    client = _StubVisionClient(response=_response(_verdict_payload(score=9.0)))
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)
    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_unknown"))
    assert result.verdict is Verdict.ERROR
    assert "does not know how to scope" in result.feedback


def test_fail_when_storefront_tree_missing(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
    artifact_dir: Path,
) -> None:
    (artifact_dir / "hydrogen" / "app").rmdir()
    (artifact_dir / "hydrogen").rmdir()
    client = _StubVisionClient(response=_response(_verdict_payload(score=9.0)))
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)
    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))
    assert result.verdict is Verdict.FAIL
    assert "storefront tree" in result.feedback


def test_error_on_unparseable_response(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    client = _StubVisionClient(response=_response(None, parse_errors=("not json",)))
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)
    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))
    assert result.verdict is Verdict.ERROR
    assert result.details["phase"] == "parse"


def test_error_on_schema_violation(
    make_fidelity_ctx: Callable[..., VerifierContext],
    data_dir: Path,
    seed_dir: Path,
) -> None:
    _seed_reference(seed_dir, "homepage_sections")
    # Missing required keys → pydantic rejects → parse ERROR.
    client = _StubVisionClient(response=_response({"overall_fidelity_score": 8.0}))
    verifier = _build_verifier(data_dir=data_dir, seed_dir=seed_dir, client=client)
    result = verifier.run(make_fidelity_ctx(selected_task_id="gen_homepage"))
    assert result.verdict is Verdict.ERROR
    assert result.details["phase"] == "parse"
