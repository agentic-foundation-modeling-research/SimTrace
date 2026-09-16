"""Unit tests for :mod:`shop_arena.gen.data_synth.navigation_ingest`.

Covers the verbatim navigation-ingest mode:

* :func:`build_navigation`: type classification per URL path prefix, same-host
  URL relativization, external-link passthrough, and nested children.
* :class:`IngestNavigationStep`: the step contract, the ``--list-steps``
  placeholder branch, an end-to-end ``run`` writing a schema-valid cache from
  a captured feed, determinism, and the missing-file error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shop_arena.gen.config import ShopGenConfig
from shop_arena.gen.data_synth import (
    IngestNavigationStep,
    Navigation,
    StageSynthError,
    build_navigation,
)
from shop_arena.gen.data_synth.navigation_ingest import RawNavigationFeed
from shop_arena.gen.steps.base import FileInput, StepContext

_BASE_URL = "https://ami-paris.com"


def _feed(
    menus: dict[str, list[dict[str, object]]], *, base_url: str = _BASE_URL
) -> RawNavigationFeed:
    """Build a :class:`RawNavigationFeed` from a plain menus dict."""
    return RawNavigationFeed.model_validate({"base_url": base_url, "menus": menus})


def _write_capture(
    seed_dir: Path,
    menus: dict[str, list[dict[str, object]]],
    *,
    base_url: str = _BASE_URL,
) -> None:
    """Write a captured ``navigation.json`` under the seed's prefetch dir."""
    prefetch = seed_dir / "artifact" / "prefetch"
    prefetch.mkdir(parents=True)
    (prefetch / "navigation.json").write_text(
        json.dumps({"base_url": base_url, "menus": menus}),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# build_navigation
# --------------------------------------------------------------------------- #


def test_build_navigation_classifies_types_by_prefix() -> None:
    feed = _feed(
        {
            "main-menu": [
                {"title": "Bags", "url": f"{_BASE_URL}/collections/bags", "children": []},
                {"title": "A Tote", "url": f"{_BASE_URL}/products/tote", "children": []},
                {"title": "About", "url": f"{_BASE_URL}/pages/about", "children": []},
                {"title": "Journal", "url": f"{_BASE_URL}/blogs/journal", "children": []},
                {"title": "Home", "url": f"{_BASE_URL}/", "children": []},
            ],
        },
    )
    nav = build_navigation(feed)
    items = nav.root["main-menu"]
    assert [i.type for i in items] == ["COLLECTION", "PRODUCT", "PAGE", "BLOG", "HTTP"]


def test_build_navigation_relativizes_same_host_urls() -> None:
    feed = _feed(
        {
            "main-menu": [
                {
                    "title": "Bags",
                    "url": f"{_BASE_URL}/collections/bags?sort=price#top",
                    "children": [],
                },
            ],
        },
    )
    nav = build_navigation(feed)
    # Scheme + host + query + fragment stripped to a bare host-relative path.
    assert nav.root["main-menu"][0].url == "/collections/bags"


def test_build_navigation_keeps_external_urls_absolute() -> None:
    feed = _feed(
        {
            "footer": [
                {
                    "title": "Instagram",
                    "url": "https://instagram.com/amiparis",
                    "children": [],
                },
            ],
        },
    )
    nav = build_navigation(feed)
    item = nav.root["footer"][0]
    assert item.url == "https://instagram.com/amiparis"
    assert item.type == "HTTP"


def test_build_navigation_recurses_children() -> None:
    feed = _feed(
        {
            "main-menu": [
                {
                    "title": "Shop",
                    "url": f"{_BASE_URL}/collections/all",
                    "children": [
                        {"title": "Hats", "url": f"{_BASE_URL}/collections/hats", "children": []},
                    ],
                },
            ],
        },
    )
    nav = build_navigation(feed)
    parent = nav.root["main-menu"][0]
    assert parent.url == "/collections/all"
    assert len(parent.children) == 1
    child = parent.children[0]
    assert child.url == "/collections/hats"
    assert child.type == "COLLECTION"


def test_build_navigation_preserves_menu_handles_and_order() -> None:
    feed = _feed(
        {
            "main-menu": [
                {"title": "A", "url": f"{_BASE_URL}/collections/a", "children": []},
                {"title": "B", "url": f"{_BASE_URL}/collections/b", "children": []},
            ],
            "footer": [{"title": "About", "url": f"{_BASE_URL}/pages/about", "children": []}],
        },
    )
    nav = build_navigation(feed)
    assert list(nav.root.keys()) == ["main-menu", "footer"]
    assert [i.title for i in nav.root["main-menu"]] == ["A", "B"]


# --------------------------------------------------------------------------- #
# IngestNavigationStep
# --------------------------------------------------------------------------- #


def test_ingest_navigation_step_contract(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    step = IngestNavigationStep(seed_dir=seed_dir)
    assert step.id == "ingest_navigation"
    assert step.phase == "data_synth"
    assert step.depends_on == []
    assert step.outputs == [Path(".shop_gen/stage_cache/navigation.json")]
    assert step.inputs == [
        FileInput(path=seed_dir / "artifact" / "prefetch" / "navigation.json"),
    ]


def test_ingest_navigation_step_list_placeholder_has_no_inputs() -> None:
    step = IngestNavigationStep()
    assert step.id == "ingest_navigation"
    assert step.inputs == []


def test_ingest_navigation_step_run_writes_valid_cache(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_capture(
        seed_dir,
        {
            "main-menu": [
                {
                    "title": "Shop",
                    "url": f"{_BASE_URL}/collections/all",
                    "children": [
                        {"title": "Hats", "url": f"{_BASE_URL}/collections/hats", "children": []},
                    ],
                },
            ],
            "footer": [{"title": "About", "url": f"{_BASE_URL}/pages/about", "children": []}],
        },
    )
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=out_dir, catalog_source="ingest")
    IngestNavigationStep(seed_dir=seed_dir).run(
        StepContext(config=config, out_dir=out_dir, runtime=None),
    )

    cache = out_dir / ".shop_gen" / "stage_cache" / "navigation.json"
    payload = json.loads(cache.read_text(encoding="utf-8"))
    nav = Navigation.model_validate(payload)
    assert set(nav.root.keys()) == {"main-menu", "footer"}
    shop = nav.root["main-menu"][0]
    assert shop.url == "/collections/all"
    assert shop.children[0].url == "/collections/hats"
    assert nav.root["footer"][0].type == "PAGE"


def test_ingest_navigation_step_run_is_deterministic(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_capture(
        seed_dir,
        {"main-menu": [{"title": "Bags", "url": f"{_BASE_URL}/collections/bags", "children": []}]},
    )
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=tmp_path / "o", catalog_source="ingest")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    IngestNavigationStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_a))
    IngestNavigationStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_b))
    rel = Path(".shop_gen") / "stage_cache" / "navigation.json"
    assert (out_a / rel).read_text(encoding="utf-8") == (out_b / rel).read_text(encoding="utf-8")


def test_ingest_navigation_step_run_missing_capture_raises(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    (seed_dir / "artifact" / "prefetch").mkdir(parents=True)
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=(seed_dir,), out_dir=out_dir, catalog_source="ingest")
    with pytest.raises(StageSynthError, match=r"re-run shop-explore"):
        IngestNavigationStep(seed_dir=seed_dir).run(
            StepContext(config=config, out_dir=out_dir),
        )


def test_ingest_navigation_step_run_without_seed_dir_raises(tmp_path: Path) -> None:
    config = ShopGenConfig(seeds=(tmp_path / "seed",), out_dir=tmp_path / "out")
    with pytest.raises(StageSynthError, match=r"no seed directory"):
        IngestNavigationStep().run(StepContext(config=config, out_dir=tmp_path / "out"))
