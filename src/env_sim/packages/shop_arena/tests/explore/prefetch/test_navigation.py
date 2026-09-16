"""Unit tests for :mod:`shop_arena.explore.prefetch.navigation`.

Two layers:

* The best-effort ``capture_navigation`` wrapper — that it writes the
  self-describing ``{base_url, menus}`` artifact on success and swallows any
  capture failure without writing (monkeypatched, no browser needed).
* The browser-side ``_EXTRACT_MENUS_JS`` extractor — exercised against a
  static HTML fixture served from a real host in a real headless Chromium,
  asserting the region split (footer vs main-menu), one-level nesting, the
  whole-document backfill that rescues headless storefronts, dedupe, and
  locale-variant collapse. Skipped when a browser cannot be launched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shop_arena.explore.prefetch import navigation as nav_mod
from shop_arena.explore.prefetch.navigation import capture_navigation

# --------------------------------------------------------------------------- #
# capture_navigation wrapper (no browser)
# --------------------------------------------------------------------------- #


def test_capture_navigation_writes_self_describing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menus: dict[str, list[dict[str, Any]]] = {
        "main-menu": [{"title": "Bags", "url": "https://s.com/collections/bags", "children": []}],
        "footer": [],
    }
    def _fake(url: str, *, timeout: float) -> dict[str, list[dict[str, Any]]]:
        del url, timeout
        return menus

    monkeypatch.setattr(nav_mod, "_extract_menus", _fake)

    capture_navigation("https://s.com", dest_dir=tmp_path)

    payload = json.loads((tmp_path / "navigation.json").read_text(encoding="utf-8"))
    assert payload == {"base_url": "https://s.com", "menus": menus}


def test_capture_navigation_swallows_failure_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(url: str, *, timeout: float) -> dict[str, Any]:
        raise RuntimeError("no browser available")

    monkeypatch.setattr(nav_mod, "_extract_menus", _boom)

    # Must not raise; the run continues even when capture is impossible.
    capture_navigation("https://s.com", dest_dir=tmp_path)

    assert not (tmp_path / "navigation.json").exists()


# --------------------------------------------------------------------------- #
# Browser-side extractor (real headless Chromium, served from a real host)
# --------------------------------------------------------------------------- #

# Models the AMI Paris headless-storefront failure mode: the <header> nav holds
# only a logo (href="/") and a cart counter, while the real menu is a separate
# hydrated container elsewhere in the DOM (here <div id="mega">) with one nested
# submenu. Body-featured product/collection links, a footer of /pages links, an
# off-host link, a duplicate, and a locale-variant mirror exercise backfill,
# region tagging, dedupe, and locale collapse.
_FIXTURE_HTML = """
<!doctype html>
<html>
  <body>
    <header>
      <nav>
        <a href="/">
          Logo
          Store
        </a>
        <a href="/cart">0</a>
      </nav>
    </header>
    <div id="mega">
      <ul>
        <li>
          <a href="https://s.com/collections/all">
            Shop
            All
          </a>
          <ul>
            <li><a href="https://s.com/collections/hats">Hats</a></li>
            <li><a href="https://s.com/collections/bags">Bags</a></li>
            <li><a href="https://s.com/collections/hats">Hats again</a></li>
          </ul>
        </li>
        <li>
          <a href="https://s.com/collections/new">
            <span>New</span>
            <script type="application/ld+json" class="js-gtm-json">
              {"event": "click_menu", "menu_level1": "New"}
            </script>
          </a>
        </li>
        <li><a href="#">Skip me</a></li>
        <li><a href="javascript:void(0)">No-op</a></li>
      </ul>
    </div>
    <section id="featured">
      <a href="https://s.com/products/tee">Featured tee</a>
      <a href="https://s.com/ko-kr/collections/new">신상품</a>
    </section>
    <footer>
      <a href="https://s.com/pages/about">About</a>
      <a href="https://s.com/pages/contact">Contact</a>
      <a href="https://instagram.com/s">Instagram</a>
    </footer>
  </body>
</html>
"""


def _extract_from_html(html: str) -> dict[str, list[dict[str, Any]]]:
    """Serve ``html`` from a real host and run the JS extractor over it.

    The extractor filters to same-host links via ``window.location.hostname``,
    so the fixture is fulfilled from ``https://s.com/`` (via ``page.route``)
    rather than ``page.set_content`` — the latter loads ``about:blank`` and
    would make every ``https://s.com/...`` link read as off-host.

    Returns the extractor's menus object. Skips the test when a browser
    cannot be launched (missing binary / sandbox).
    """
    playwright_api = pytest.importorskip("playwright.sync_api")
    try:
        with playwright_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()

                def _fulfill(route: Any) -> None:
                    route.fulfill(status=200, content_type="text/html", body=html)

                page.route("**/*", _fulfill)
                page.goto("https://s.com/")
                result: dict[str, list[dict[str, Any]]] = page.evaluate(
                    nav_mod._EXTRACT_MENUS_JS,  # pyright: ignore[reportPrivateUsage]
                )
                return result
            finally:
                browser.close()
    except Exception as exc:
        pytest.skip(f"headless chromium unavailable: {exc}")


def test_extractor_captures_hydrated_menu_outside_header() -> None:
    """The real menu lives outside <header>; nesting + backfill still find it."""
    menus = _extract_from_html(_FIXTURE_HTML)
    main = menus["main-menu"]
    by_url = {i["url"]: i for i in main}

    # The mega-menu's structural walk yields "Shop" with one level of children.
    shop = by_url["https://s.com/collections/all"]
    assert shop["title"] == "Shop All"  # multi-line anchor text collapsed
    assert [c["url"] for c in shop["children"]] == [
        "https://s.com/collections/hats",
        "https://s.com/collections/bags",
    ]  # duplicate "Hats" href collapsed

    # Sibling menu item and the featured product are captured (structural +
    # backfill), all tagged main-menu since none sit inside <footer>.
    assert "https://s.com/collections/new" in by_url
    assert "https://s.com/products/tee" in by_url

    # The in-anchor GTM <script> JSON must not leak into the display title.
    assert by_url["https://s.com/collections/new"]["title"] == "New"


def test_extractor_skips_non_navigational_and_self_links() -> None:
    """#/javascript: hrefs and the homepage self-link (logo href="/") are dropped."""
    menus = _extract_from_html(_FIXTURE_HTML)
    urls = {i["url"] for i in menus["main-menu"]}

    assert "#" not in urls
    assert not any(u.startswith("javascript:") for u in urls)
    # The logo links back to "/", i.e. the captured page itself — a self-link,
    # so it is not recorded as a navigation target.
    assert "https://s.com/" not in urls
    assert "/" not in urls


def test_extractor_collapses_locale_variants() -> None:
    """A /ko-kr/ mirror of an existing path does not appear twice."""
    menus = _extract_from_html(_FIXTURE_HTML)
    urls = {i["url"] for i in menus["main-menu"]}
    # The ko-kr collections/new mirrors the canonical /collections/new already
    # captured, so the locale-collapsed dedup drops it.
    assert "https://s.com/ko-kr/collections/new" not in urls


def test_extractor_splits_footer_region_and_drops_off_host() -> None:
    """Same-host links inside <footer> form the footer menu; off-host dropped."""
    menus = _extract_from_html(_FIXTURE_HTML)
    footer_urls = {i["url"] for i in menus["footer"]}
    assert footer_urls == {
        "https://s.com/pages/about",
        "https://s.com/pages/contact",
    }  # the off-host instagram.com link is filtered out
    # Footer /pages links must not leak into main-menu.
    main_urls = {i["url"] for i in menus["main-menu"]}
    assert "https://s.com/pages/about" not in main_urls
