"""Headless-browser capture of the storefront's real navigation menus.

The deterministic HTTP prefetch (:mod:`shop_arena.explore.prefetch.runner`)
is intentionally browser-free — it fetches Shopify's JSON feeds and static
pages over ``httpx`` and never executes page scripts. Navigation menus,
however, are rendered markup: the header/primary nav and the footer only
exist as DOM once the storefront's HTML loads. This module fills that gap
with a small, best-effort Playwright capture that records the site's menus
verbatim into ``dest_dir/navigation.json``.

The capture is deliberately "dumb": it records each menu item's display
title, its absolute ``href``, and one level of nested children. It performs
**no** classification of link targets — mapping the raw menu into the
SandboxShop ``Navigation`` schema (relativizing URLs, classifying item
types, recursing children) is the job of
:mod:`shop_arena.gen.data_synth.navigation_ingest`, which consumes this
artifact in ``catalog_source='ingest'`` mode. Keeping the browser-dependent
scraping here and the pure schema mapping in ``gen`` mirrors the
``products.json`` / ``ingest_catalog`` split.

Raw artifact shape (``navigation.json``). The captured menus are nested
under ``menus`` alongside the ``base_url`` they were captured from, so the
downstream ``gen`` mapper is self-contained (it needs the store host to
relativize same-host links but has no storefront URL in its own config)::

    {
      "base_url": "https://site.com",
      "menus": {
        "footer": [
          {"title": "About", "url": "https://site.com/pages/about", "children": []}
        ],
        "main-menu": [
          {"title": "Shop", "url": "https://site.com/collections/all", "children": [
            {"title": "Hats", "url": "https://site.com/collections/hats", "children": []}
          ]}
        ]
      }
    }

The function is best-effort: any failure (Playwright missing, no browser
binary, navigation timeout, or an unexpected DOM) is logged and swallowed
so it never aborts an explore run. When capture fails the file is simply
not written; the downstream ``ingest_navigation`` step then fails loudly
with a "re-run shop-explore" message.

Module is import-safe: no I/O, no env reads, no side effects at import.
``playwright`` is imported lazily inside :func:`capture_navigation` so the
package stays importable on machines without the browser installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

_log = logging.getLogger(__name__)

DEFAULT_CAPTURE_TIMEOUT_SECONDS: Final[float] = 30.0
"""Per-page navigation timeout for :func:`capture_navigation`."""

_OUT_FILENAME: Final[str] = "navigation.json"
"""Artifact written under ``dest_dir`` on a successful capture."""

_EXTRACT_MENUS_JS: Final[str] = r"""
() => {
  const here = window.location.href;
  const host = window.location.hostname;

  // Skip is decided on the *raw* href attribute: a bare "#" or a
  // "javascript:"/"mailto:"/"tel:" scheme resolves to a non-navigational
  // absolute URL, so testing anchor.href (absolute) would let "#" through.
  const skipRaw = (raw) =>
    !raw || raw.startsWith('#') || raw.startsWith('javascript:') ||
    raw.startsWith('mailto:') || raw.startsWith('tel:');

  // A same-host outbound navigation target: http(s), same hostname, and not a
  // self-link back to the current page. `anchor.href` is DOM-resolved (absolute).
  const sameHostOuter = (anchor) => {
    if (skipRaw(anchor.getAttribute('href'))) return null;
    let url;
    try { url = new URL(anchor.href, here); } catch (_e) { return null; }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    if (url.hostname !== host) return null;
    const abs = url.href;
    if (abs === here) return null;
    return abs;
  };

  // Read the display label from text nodes only. Storefronts commonly embed a
  // <script type="application/ld+json"> analytics blob *inside* the anchor
  // (e.g. AMI's GTM "js-gtm-json"), which anchor.textContent would concatenate
  // into the title. Clone + strip non-text elements first, then normalize
  // whitespace. Cloning (not innerText) keeps labels for menu items that are
  // display:none until hover — innerText returns "" for unrendered nodes.
  const title = (anchor) => {
    const clone = anchor.cloneNode(true);
    for (const el of clone.querySelectorAll('script, style, template, noscript')) el.remove();
    return (clone.textContent || '').replace(/\s+/g, ' ').trim();
  };

  // A locale prefix mirrors the canonical path under /<lang>/ or /<lang>-<region>/
  // (e.g. /ko-kr/pages/about, /fr/collections/x). Collapsing to the canonical
  // key de-dupes the country-switcher explosion (~200 links) against the menu.
  const localeKey = (abs) => {
    const u = new URL(abs);
    const path = u.pathname.replace(/^\/([a-z]{2})(-[a-z]{2})?(?=\/)/i, '');
    return u.hostname + (path || '/') + u.search;
  };

  // Walk `root` one level deep: a top-level <li>'s own anchor is an item;
  // anchors inside that <li>'s nested list/submenu become its children. Menus
  // without <li> structure fall back to the root's descendant anchors (flat).
  // `seen` is the run-wide locale-collapsed key set so items already captured
  // by another menu (or the backfill) are not duplicated.
  const collect = (root, seen) => {
    if (!root) return [];
    const items = [];
    // Only treat an <li> as top-level when it is not itself nested inside
    // another <li>'s submenu (avoids double-listing children).
    const isNested = (li) => li.parentElement && li.parentElement.closest('li') !== null;
    for (const li of root.querySelectorAll(':scope li')) {
      if (isNested(li)) continue;
      const anchor = li.querySelector(':scope > a') || li.querySelector('a');
      if (!anchor) continue;
      const abs = sameHostOuter(anchor);
      const label = title(anchor);
      if (!abs || !label) continue;
      const key = localeKey(abs);
      if (seen.has(key)) continue;
      seen.add(key);
      const children = [];
      for (const sub of li.querySelectorAll(':scope ul a, :scope ol a')) {
        const subAbs = sameHostOuter(sub);
        const subLabel = title(sub);
        if (!subAbs || !subLabel) continue;
        const subKey = localeKey(subAbs);
        if (seen.has(subKey)) continue;
        seen.add(subKey);
        children.push({title: subLabel, url: subAbs, children: []});
      }
      items.push({title: label, url: abs, children});
    }
    return items;
  };

  const footerEl = document.querySelector('footer');
  const seen = new Set();

  // Footer first so footer links claim the footer menu before the main-menu
  // structural walk / backfill can absorb them. Then walk the whole body for
  // menu structure: the primary menu may live anywhere (on headless
  // storefronts it is not under a recognizable <header>/<nav>), so scoping to
  // one container would miss it. Footer <li>s are re-visited here but skipped
  // via `seen`.
  const footer = collect(footerEl, seen);
  const main = collect(document.body, seen);

  // Completeness backfill: any same-host outbound link not yet captured by the
  // structural walk is appended flat, tagged by region (inside <footer> or not).
  // This is what rescues headless storefronts whose real menu is not under a
  // recognizable nav container (the AMI failure mode).
  for (const anchor of document.querySelectorAll('a[href]')) {
    const abs = sameHostOuter(anchor);
    const label = title(anchor);
    if (!abs || !label) continue;
    const key = localeKey(abs);
    if (seen.has(key)) continue;
    seen.add(key);
    const item = {title: label, url: abs, children: []};
    if (footerEl && footerEl.contains(anchor)) footer.push(item);
    else main.push(item);
  }

  return {'main-menu': main, 'footer': footer};
}
"""
"""Browser-side extractor: the storefront's real navigation map.

Captures every same-host outbound link on the *hydrated* homepage, tagged by
DOM region — links inside ``<footer>`` form the ``footer`` menu, everything
else forms ``main-menu``. A structural walk over the footer, then the whole
``<body>``, recovers one level of submenu nesting from list markup wherever
the menu lives; a whole-document backfill then appends any same-host link the
structural walk missed, so client-rendered menus that live outside a
recognizable nav container are still captured in full.

Empty / ``#`` / ``javascript:`` / ``mailto:`` / ``tel:`` hrefs are skipped
(decided on the raw attribute). Titles are read from text nodes only —
``<script>``/``<style>`` descendants (e.g. an in-anchor analytics JSON blob)
are stripped before whitespace normalization. Links are de-duplicated by a
locale-collapsed key, so alternate-locale mirrors of the same path
(``/ko-kr/…``, ``/fr/…``) do not flood the menu.
"""


def capture_navigation(
    url: str,
    *,
    dest_dir: Path,
    timeout: float = DEFAULT_CAPTURE_TIMEOUT_SECONDS,
) -> None:
    """Capture the storefront's header + footer menus into ``navigation.json``.

    Launches a headless Chromium via Playwright, loads ``url``, extracts the
    primary navigation and footer menus (titles + absolute hrefs + one level
    of nested children), and writes them to ``dest_dir/navigation.json`` with
    the canonical ``json.dumps(indent=2, sort_keys=True)`` convention.

    Best-effort: any failure — Playwright not installed, no browser binary,
    a navigation timeout, or an unexpected DOM — is logged and swallowed. On
    failure nothing is written, so the absence of the file is the signal that
    capture did not run.

    Args:
        url: Storefront base URL to load. Must include scheme + host.
        dest_dir: Directory to write ``navigation.json`` into. Created if
            missing.
        timeout: Per-page navigation timeout in seconds.
    """
    try:
        menus = _extract_menus(url, timeout=timeout)
    except Exception as exc:
        _log.warning("navigation capture failed for %s: %s: %s", url, type(exc).__name__, exc)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {"base_url": url, "menus": menus}
    (dest_dir / _OUT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log.info(
        "captured navigation from %s: main-menu=%d, footer=%d",
        url,
        len(menus.get("main-menu", [])),
        len(menus.get("footer", [])),
    )


def _extract_menus(url: str, *, timeout: float) -> dict[str, list[dict[str, Any]]]:
    """Load ``url`` in headless Chromium and return the raw menus object.

    Args:
        url: Storefront base URL.
        timeout: Per-page navigation timeout in seconds.

    Returns:
        The ``{"main-menu": [...], "footer": [...]}`` object produced by
        :data:`_EXTRACT_MENUS_JS`.

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
            result: dict[str, list[dict[str, Any]]] = page.evaluate(_EXTRACT_MENUS_JS)
            return result
        finally:
            browser.close()


__all__ = [
    "DEFAULT_CAPTURE_TIMEOUT_SECONDS",
    "capture_navigation",
]
