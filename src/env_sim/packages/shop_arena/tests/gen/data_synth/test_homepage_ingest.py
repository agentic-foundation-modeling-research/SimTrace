"""Unit tests for :mod:`shop_arena.gen.data_synth.homepage_ingest`.

Covers the verbatim site-level hero/banner ingest:

* :func:`build_homepage`: verbatim URL hotlinking, ``alt=None`` brand-leak
  safety, missing-dimension defaults, banner ordering, and the null-hero case.
* :class:`IngestHomepageStep`: the step contract (inputs declared only when the
  capture file exists), an end-to-end ``run`` writing a schema-valid
  ``homepage.json``, determinism, and the best-effort empty-homepage fallbacks
  (no seed, missing capture, malformed capture).
"""

from __future__ import annotations

import json
from pathlib import Path

from shop_arena.gen.config import ShopGenConfig
from shop_arena.gen.data_synth import (
    Homepage,
    IngestHomepageStep,
    RawHeroFeed,
    build_homepage,
)
from shop_arena.gen.steps.base import FileInput, StepContext

_HERO_URL = "https://cdn.example.com/hero-spring.jpg"
_BANNER_URL = "https://cdn.example.com/promo-sale.jpg"
_PREFETCH_HERO = Path("artifact") / "prefetch" / "hero.json"


def _write_capture(seed_dir: Path, payload: dict[str, object]) -> None:
    """Write a captured ``hero.json`` under the seed's prefetch dir."""
    prefetch = seed_dir / "artifact" / "prefetch"
    prefetch.mkdir(parents=True)
    (prefetch / "hero.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# build_homepage
# --------------------------------------------------------------------------- #


def test_build_homepage_hotlinks_url_verbatim_and_drops_alt() -> None:
    feed = RawHeroFeed.model_validate(
        {
            "base_url": "https://ami-paris.com",
            "hero": {"url": _HERO_URL, "width": 2400, "height": 1200, "alt": "Spring campaign"},
            "banners": [],
        },
    )
    homepage = build_homepage(feed)
    assert homepage.hero is not None
    assert homepage.hero.src == _HERO_URL
    assert homepage.hero.alt is None
    assert (homepage.hero.width, homepage.hero.height) == (2400, 1200)
    assert homepage.hero.position == 1


def test_build_homepage_defaults_missing_dimensions_to_zero() -> None:
    feed = RawHeroFeed.model_validate({"hero": {"url": _HERO_URL}, "banners": []})
    assert feed.hero is not None
    homepage = build_homepage(feed)
    assert homepage.hero is not None
    assert (homepage.hero.width, homepage.hero.height) == (0, 0)


def test_build_homepage_null_hero_and_empty_banners() -> None:
    homepage = build_homepage(RawHeroFeed())
    assert homepage.hero is None
    assert homepage.banners == []


def test_build_homepage_preserves_banner_order_and_positions() -> None:
    feed = RawHeroFeed.model_validate(
        {
            "hero": None,
            "banners": [
                {"url": "https://cdn.example.com/a.jpg", "width": 1600, "height": 600},
                {"url": "https://cdn.example.com/b.jpg", "width": 1600, "height": 600},
            ],
        },
    )
    homepage = build_homepage(feed)
    assert [b.src for b in homepage.banners] == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.jpg",
    ]
    assert [b.position for b in homepage.banners] == [1, 2]
    assert all(b.alt is None for b in homepage.banners)


# --------------------------------------------------------------------------- #
# IngestHomepageStep — contract
# --------------------------------------------------------------------------- #


def test_ingest_homepage_step_contract_declares_input_when_capture_present(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed"
    _write_capture(seed_dir, {"hero": None, "banners": []})
    step = IngestHomepageStep(seed_dir=seed_dir)
    assert step.id == "ingest_homepage"
    assert step.phase == "data_synth"
    assert step.depends_on == []
    assert step.outputs == [Path("data/homepage.json")]
    assert step.inputs == [FileInput(path=seed_dir / _PREFETCH_HERO)]


def test_ingest_homepage_step_omits_input_when_capture_absent(tmp_path: Path) -> None:
    """A missing best-effort capture must not be declared (would break fingerprinting)."""
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    step = IngestHomepageStep(seed_dir=seed_dir)
    assert step.inputs == []


def test_ingest_homepage_step_list_placeholder_has_no_inputs() -> None:
    step = IngestHomepageStep()
    assert step.id == "ingest_homepage"
    assert step.inputs == []


# --------------------------------------------------------------------------- #
# IngestHomepageStep — run
# --------------------------------------------------------------------------- #


def test_ingest_homepage_step_run_writes_valid_homepage(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_capture(
        seed_dir,
        {
            "hero": {"url": _HERO_URL, "width": 2400, "height": 1200, "alt": "hi"},
            "banners": [{"url": _BANNER_URL, "width": 1600, "height": 600}],
        },
    )
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=out_dir)
    IngestHomepageStep(seed_dir=seed_dir).run(
        StepContext(config=config, out_dir=out_dir, runtime=None),
    )

    payload = json.loads((out_dir / "data" / "homepage.json").read_text(encoding="utf-8"))
    homepage = Homepage.model_validate(payload)
    assert homepage.hero is not None
    assert homepage.hero.src == _HERO_URL
    assert homepage.hero.alt is None
    assert [b.src for b in homepage.banners] == [_BANNER_URL]


def test_ingest_homepage_step_run_is_deterministic(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_capture(
        seed_dir,
        {"hero": {"url": _HERO_URL, "width": 2400, "height": 1200}, "banners": []},
    )
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=tmp_path / "o")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    IngestHomepageStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_a))
    IngestHomepageStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_b))
    rel = Path("data") / "homepage.json"
    assert (out_a / rel).read_text(encoding="utf-8") == (out_b / rel).read_text(encoding="utf-8")


def test_ingest_homepage_step_run_without_seed_dir_writes_empty(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=(tmp_path / "seed",), out_dir=out_dir)
    IngestHomepageStep().run(StepContext(config=config, out_dir=out_dir))

    payload = json.loads((out_dir / "data" / "homepage.json").read_text(encoding="utf-8"))
    assert payload == {"hero": None, "banners": []}


def test_ingest_homepage_step_run_missing_capture_writes_empty(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    (seed_dir / "artifact" / "prefetch").mkdir(parents=True)
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=out_dir)
    IngestHomepageStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_dir))

    payload = json.loads((out_dir / "data" / "homepage.json").read_text(encoding="utf-8"))
    assert payload == {"hero": None, "banners": []}


def test_ingest_homepage_step_run_malformed_capture_writes_empty(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_capture(seed_dir, {"hero": "not-an-object"})
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=out_dir)
    IngestHomepageStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_dir))

    payload = json.loads((out_dir / "data" / "homepage.json").read_text(encoding="utf-8"))
    assert payload == {"hero": None, "banners": []}
