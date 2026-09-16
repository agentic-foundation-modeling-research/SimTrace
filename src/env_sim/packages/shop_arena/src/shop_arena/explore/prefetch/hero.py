"""Headless-browser capture of the storefront's real hero / banner imagery.

The deterministic HTTP prefetch (:mod:`shop_arena.explore.prefetch.runner`)
is intentionally browser-free — it fetches Shopify's JSON feeds and static
pages over ``httpx`` and never executes page scripts, so it never learns the
URLs of the editorial hero and promo-banner images that make a storefront's
homepage look the way it does. Those images only exist as ``<img>`` sources
and CSS ``background-image`` URLs once the homepage's HTML and scripts run.
This module fills that gap with a small, best-effort Playwright capture that
records those image URLs verbatim into ``dest_dir/hero.json``.

The capture is deliberately "dumb": it records the single most prominent
above-the-fold image (the hero) plus a capped, ordered list of large
full-bleed image/background blocks below it (promo banners). It downloads
**no** bytes and performs **no** classification beyond size/position
heuristics — turning the captured URLs into the SandboxShop ``homepage.json``
is the job of :mod:`shop_arena.gen.data_synth.hero_ingest`, which consumes
this artifact. Keeping the browser-dependent scraping here and the pure
schema mapping in ``gen`` mirrors the ``navigation.json`` /
``ingest_navigation`` split.

The store logo is deliberately excluded: images inside ``<header>``, elements
whose class hints at a logo, and inline ``<svg>`` are skipped, as are images
below a minimum rendered-area threshold.

Raw artifact shape (``hero.json``). Image URLs are absolute (the DOM resolves
them for us) and each entry carries the rendered pixel dimensions so the
downstream mapper does not need to re-decode the bytes::

    {
      "base_url": "https://site.com",
      "hero": {
        "url": "https://cdn.site.com/hero.jpg",
        "width": 2400, "height": 1200, "alt": "Spring campaign"
      },
      "banners": [
        {"url": "https://cdn.site.com/promo.jpg", "width": 1600, "height": 600, "alt": null}
      ]
    }

``hero`` is ``null`` when no above-the-fold image clears the thresholds;
``banners`` is ``[]`` when none are found.

The function is best-effort: any failure (Playwright missing, no browser
binary, navigation timeout, or an unexpected DOM) is logged and swallowed so
it never aborts an explore run. When capture fails the file is simply not
written; the downstream ``hero_ingest`` step then falls back to an empty
homepage.

Module is import-safe: no I/O, no env reads, no side effects at import.
``playwright`` is imported lazily inside :func:`capture_hero` so the package
stays importable on machines without the browser installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

_log = logging.getLogger(__name__)

DEFAULT_CAPTURE_TIMEOUT_SECONDS: Final[float] = 30.0
"""Per-page navigation timeout for :func:`capture_hero`."""

_OUT_FILENAME: Final[str] = "hero.json"
"""Artifact written under ``dest_dir`` on a successful capture."""

_EXTRACT_HERO_JS: Final[str] = r"""
() => {
  // Minimum rendered area (px^2) for an element to count as a hero/banner.
  // Filters out logos, icons, thumbnails, and decorative chrome.
  const MIN_AREA = 90000;          // ~ 300 x 300
  // The hero must start near the top of the page.
  const HERO_MAX_TOP = 900;
  const MAX_BANNERS = 4;

  const absolutize = (raw) => {
    if (!raw) return null;
    try { return new URL(raw, document.baseURI).href; } catch (e) { return null; }
  };

  // Pull a usable image URL from an element: an <img> currentSrc/src, or the
  // first url(...) in its computed background-image.
  const urlFor = (el) => {
    if (el.tagName === 'IMG') return absolutize(el.currentSrc || el.getAttribute('src'));
    const bg = window.getComputedStyle(el).backgroundImage || '';
    const m = bg.match(/url\((['"]?)(.*?)\1\)/);
    return m ? absolutize(m[2]) : null;
  };

  // Skip logos and chrome: anything inside a header, anything whose class/id
  // looks like a logo, and inline SVG.
  const isChrome = (el) => {
    if (el.closest('header')) return true;
    if (el.closest('svg')) return true;
    const hint = ((el.className && el.className.baseVal !== undefined
      ? el.className.baseVal : el.className) || '') + ' ' + (el.id || '');
    return /logo|icon|payment|social/i.test(hint);
  };

  // Candidate elements: <img> plus elements carrying a background-image.
  const candidates = [];
  const seenUrls = new Set();
  const push = (el) => {
    if (isChrome(el)) return;
    const url = urlFor(el);
    if (!url || seenUrls.has(url)) return;
    const rect = el.getBoundingClientRect();
    const rw = Math.round(rect.width);
    const rh = Math.round(rect.height);
    const w = el.tagName === 'IMG' ? (el.naturalWidth || rw) : rw;
    const h = el.tagName === 'IMG' ? (el.naturalHeight || rh) : rh;
    const area = rw * rh;
    if (area < MIN_AREA) return;
    const top = rect.top + window.scrollY;
    seenUrls.add(url);
    candidates.push({
      url,
      width: w || null,
      height: h || null,
      alt: el.tagName === 'IMG' ? ((el.getAttribute('alt') || '').trim() || null) : null,
      top,
      area,
    });
  };

  for (const el of document.querySelectorAll('img')) push(el);
  for (const el of document.querySelectorAll('*')) {
    const bg = window.getComputedStyle(el).backgroundImage || '';
    if (bg && bg !== 'none' && bg.includes('url(')) push(el);
  }

  // Hero: the largest above-the-fold candidate.
  const heroPool = candidates
    .filter((c) => c.top <= HERO_MAX_TOP)
    .sort((a, b) => b.area - a.area);
  const hero = heroPool.length ? heroPool[0] : null;

  // Banners: remaining candidates below the hero, in DOM/scroll order, capped.
  const heroUrl = hero ? hero.url : null;
  const banners = candidates
    .filter((c) => c.url !== heroUrl)
    .sort((a, b) => a.top - b.top)
    .slice(0, MAX_BANNERS);

  const strip = (c) => c === null ? null
    : {url: c.url, width: c.width, height: c.height, alt: c.alt};
  return {hero: strip(hero), banners: banners.map(strip)};
}
"""
"""Browser-side extractor: one hero + up to four promo banners.

Collects every ``<img>`` and every element carrying a computed
``background-image``, absolutizes its URL, and filters out logos/chrome
(anything inside ``<header>``, class/id hints of ``logo``/``icon``/etc., and
inline ``<svg>``) plus anything below a minimum rendered area. The hero is the
largest above-the-fold survivor; banners are the remaining survivors in
scroll order, de-duplicated by URL and capped.
"""


def capture_hero(
    url: str,
    *,
    dest_dir: Path,
    timeout: float = DEFAULT_CAPTURE_TIMEOUT_SECONDS,
) -> None:
    """Capture the storefront's hero + promo-banner image URLs into ``hero.json``.

    Launches a headless Chromium via Playwright, loads ``url``, extracts the
    most prominent above-the-fold image (the hero) and a capped, ordered list
    of large full-bleed image/background blocks (promo banners), and writes
    them to ``dest_dir/hero.json`` with the canonical
    ``json.dumps(indent=2, sort_keys=True)`` convention. Only absolute URLs
    and rendered pixel dimensions are recorded — no bytes are downloaded.

    Best-effort: any failure — Playwright not installed, no browser binary, a
    navigation timeout, or an unexpected DOM — is logged and swallowed. On
    failure nothing is written, so the absence of the file is the signal that
    capture did not run.

    Args:
        url: Storefront base URL to load. Must include scheme + host.
        dest_dir: Directory to write ``hero.json`` into. Created if missing.
        timeout: Per-page navigation timeout in seconds.
    """
    try:
        media = _extract_hero(url, timeout=timeout)
    except Exception as exc:
        _log.warning("hero capture failed for %s: %s: %s", url, type(exc).__name__, exc)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {"base_url": url, "hero": media["hero"], "banners": media["banners"]}
    (dest_dir / _OUT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log.info(
        "captured hero media from %s: hero=%s, banners=%d",
        url,
        "yes" if media["hero"] is not None else "no",
        len(media["banners"]),
    )


def _extract_hero(url: str, *, timeout: float) -> dict[str, Any]:
    """Load ``url`` in headless Chromium and return the raw hero/banner object.

    Args:
        url: Storefront base URL.
        timeout: Per-page navigation timeout in seconds.

    Returns:
        The ``{"hero": {...} | None, "banners": [...]}`` object produced by
        :data:`_EXTRACT_HERO_JS`.

    Raises:
        Exception: Any Playwright / navigation / evaluation error. Callers
            treat every failure as "capture unavailable".
    """
    # Imported lazily so this module stays import-safe and cheap: Playwright is
    # a heavy dependency only needed when a capture actually runs, and a missing
    # install surfaces here as the swallowed "capture unavailable" path.
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    timeout_ms = int(timeout * 1000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except Exception as exc:
                _log.debug("networkidle wait timed out for %s: %s", url, exc)
            result: dict[str, Any] = page.evaluate(_EXTRACT_HERO_JS)
            return result
        finally:
            browser.close()


__all__ = [
    "DEFAULT_CAPTURE_TIMEOUT_SECONDS",
    "capture_hero",
]
