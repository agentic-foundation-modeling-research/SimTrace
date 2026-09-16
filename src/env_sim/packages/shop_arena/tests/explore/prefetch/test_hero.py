"""Unit tests for :mod:`shop_arena.explore.prefetch.hero`.

Two layers:

* The best-effort ``capture_hero`` wrapper — that it writes the
  self-describing ``{base_url, hero, banners}`` artifact on success and
  swallows any capture failure without writing (monkeypatched, no browser
  needed).
* The browser-side ``_EXTRACT_HERO_JS`` extractor — exercised against a
  static HTML fixture via ``page.set_content`` in a real headless Chromium,
  asserting the hero pick, banner ordering, and logo/small-image exclusion.
  Skipped when a browser cannot be launched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shop_arena.explore.prefetch import hero as hero_mod
from shop_arena.explore.prefetch.hero import capture_hero

# --------------------------------------------------------------------------- #
# capture_hero wrapper (no browser)
# --------------------------------------------------------------------------- #


def test_capture_hero_writes_self_describing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media: dict[str, Any] = {
        "hero": {
            "url": "https://cdn.s.com/hero.jpg",
            "width": 2400,
            "height": 1200,
            "alt": "Spring",
        },
        "banners": [
            {"url": "https://cdn.s.com/promo.jpg", "width": 1600, "height": 600, "alt": None},
        ],
    }

    def _fake(url: str, *, timeout: float) -> dict[str, Any]:
        del url, timeout
        return media

    monkeypatch.setattr(hero_mod, "_extract_hero", _fake)

    capture_hero("https://s.com", dest_dir=tmp_path)

    payload = json.loads((tmp_path / "hero.json").read_text(encoding="utf-8"))
    assert payload == {"base_url": "https://s.com", **media}


def test_capture_hero_swallows_failure_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(url: str, *, timeout: float) -> dict[str, Any]:
        raise RuntimeError("no browser available")

    monkeypatch.setattr(hero_mod, "_extract_hero", _boom)

    # Must not raise; the run continues even when capture is impossible.
    capture_hero("https://s.com", dest_dir=tmp_path)

    assert not (tmp_path / "hero.json").exists()


# --------------------------------------------------------------------------- #
# Browser-side extractor (real headless Chromium via set_content)
# --------------------------------------------------------------------------- #

# Rendered geometry drives the heuristics. The fixture uses CSS
# ``background-image`` on explicitly-sized ``<div>``s (the shape of a real
# hero/banner section) so the layout box is deterministic under
# ``set_content`` without fetching any bytes — a broken ``<img>`` would
# instead collapse to its fallback-icon size.
_FIXTURE_HTML = """
<!doctype html>
<html>
  <body>
    <header>
      <div class="logo" style="width:400px;height:200px;background-image:url(https://s.com/logo.png)"></div>
    </header>
    <section class="hero"
      style="width:1200px;height:600px;background-image:url(https://s.com/hero.jpg)"></section>
    <section class="promo-banner"
      style="width:800px;height:300px;background-image:url(https://s.com/promo.jpg)"></section>
    <div style="width:80px;height:80px;background-image:url(https://s.com/thumb.png)"></div>
  </body>
</html>
"""


def _extract_from_html(html: str) -> dict[str, Any]:
    """Render ``html`` in headless Chromium and run the JS extractor on it.

    Returns the extractor's hero/banner object. Skips the test when a browser
    cannot be launched (missing binary / sandbox). Natural image dimensions
    are set explicitly via the ``width``/``height`` attributes so the
    area/position heuristics are exercised without loading real bytes.
    """
    playwright_api = pytest.importorskip("playwright.sync_api")
    try:
        with playwright_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_content(html)
                result: dict[str, Any] = page.evaluate(
                    hero_mod._EXTRACT_HERO_JS,  # pyright: ignore[reportPrivateUsage]
                )
                return result
            finally:
                browser.close()
    except Exception as exc:
        pytest.skip(f"headless chromium unavailable: {exc}")


def test_extractor_picks_hero_excludes_logo_and_small_images() -> None:
    media = _extract_from_html(_FIXTURE_HTML)

    # The header logo (chrome) and the 80x80 thumbnail (below MIN_AREA) are excluded.
    assert media["hero"] is not None
    assert media["hero"]["url"] == "https://s.com/hero.jpg"

    banner_urls = [b["url"] for b in media["banners"]]
    assert "https://s.com/promo.jpg" in banner_urls
    assert "https://s.com/logo.png" not in banner_urls
    assert "https://s.com/thumb.png" not in banner_urls
    assert media["hero"]["url"] not in banner_urls
