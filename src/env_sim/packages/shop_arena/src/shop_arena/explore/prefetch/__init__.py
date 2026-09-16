"""Deterministic HTTP prefetch step for ``shop_arena.explore``.

Implements §5.9 of the ShopExplore spec
(``docs/specs/shop_arena/shop_arena.explore.md``): a small, fixed-URL HTTP
fetch that seeds the harness ``run_dir/artifact/prefetch/`` with enough
storefront context for the planner to reason about coverage.

No LLM. No browsing. No crawler. The agents do all further interaction
with the shop through the playwright skill.

Public surface (re-exported here so callers keep using
``shop_arena.explore.prefetch.X``):

* :class:`PrefetchEntry` — per-URL outcome record.
* :class:`PrefetchResult` — summary written to ``prefetch.json``.
* :class:`ShopUnreachableError` — raised on bot-block / fatal index.
* :func:`run` — fetch a storefront into ``dest_dir``.
* :func:`capture_navigation` — best-effort headless-browser capture of the
  storefront's header + footer menus into ``navigation.json``.
* :func:`capture_hero` — best-effort headless-browser capture of the
  storefront's hero + promo-banner image URLs into ``hero.json``.
* :data:`DEFAULT_USER_AGENT`, :data:`DEFAULT_RATE_LIMIT_MS`,
  :data:`DEFAULT_TIMEOUT_SECONDS` — defaults for :func:`run`.

Submodules:

* :mod:`shop_arena.explore.prefetch.models` — typed result records and the
  :class:`ShopUnreachableError` exception.
* :mod:`shop_arena.explore.prefetch.runner` — the fetch plan, bot-block
  detection, and the :func:`run` entrypoint.
* :mod:`shop_arena.explore.prefetch.navigation` — headless-browser menu
  capture (:func:`capture_navigation`).
* :mod:`shop_arena.explore.prefetch.hero` — headless-browser hero/banner
  image-URL capture (:func:`capture_hero`).
"""

from __future__ import annotations

from shop_arena.explore.prefetch.hero import capture_hero
from shop_arena.explore.prefetch.models import (
    PrefetchEntry,
    PrefetchResult,
    ShopUnreachableError,
)
from shop_arena.explore.prefetch.navigation import (
    DEFAULT_CAPTURE_TIMEOUT_SECONDS,
    capture_navigation,
)
from shop_arena.explore.prefetch.runner import (
    DEFAULT_RATE_LIMIT_MS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    PRODUCTS_MAX_PAGES,
    PRODUCTS_PAGE_LIMIT,
    run,
)

__all__ = [
    "DEFAULT_CAPTURE_TIMEOUT_SECONDS",
    "DEFAULT_RATE_LIMIT_MS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "PRODUCTS_MAX_PAGES",
    "PRODUCTS_PAGE_LIMIT",
    "PrefetchEntry",
    "PrefetchResult",
    "ShopUnreachableError",
    "capture_hero",
    "capture_navigation",
    "run",
]
