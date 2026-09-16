"""``ingest_navigation`` — verbatim navigation ingest from a real storefront.

Alternative to the LLM navigation-synthesis step
(:mod:`shop_arena.gen.data_synth.navigation`). Instead of inventing menus
from the synthesized collections/pages/capabilities, this step reads the
storefront's real header + footer menus that ``shop_arena.explore`` captured
with a headless browser at ``<seed>/artifact/prefetch/navigation.json`` and
maps them **verbatim** into the SandboxShop :class:`Navigation` schema — the
twin shop's navigation then mirrors the source site's own menu structure.

The raw capture (see
:func:`shop_arena.explore.prefetch.capture_navigation`) records each menu
item's display title, its absolute ``href``, and one level of nested
children, with no interpretation of the link targets. This module is the
pure, browser-free mapper that turns that raw shape into schema records:

* **Relativize** each ``url`` to a host-relative path (``/collections/hats``)
  when it points at the store host; keep off-host URLs absolute.
* **Classify** each item's :data:`~shop_arena.gen.data_synth.schema.NavigationItemType`
  by path prefix (``/collections/`` → ``COLLECTION`` …), defaulting to
  ``HTTP`` for anything else (including external links).
* **Recurse** into children.

Unlike :mod:`shop_arena.gen.data_synth.navigation`, this step does **not**
enforce "every collection is reachable from ``main-menu``": the copy is
verbatim, and a real storefront need not link every collection from its
header. That reachability invariant is specific to LLM synthesis, where the
model must not silently drop a synthesized category.

The step writes the same cache path the synth step writes
(``.shop_gen/stage_cache/navigation.json``), so the terminal
``assemble_data`` step re-reads and re-validates it unchanged — it is a
drop-in replacement in ``catalog_source='ingest'`` mode.

Module is import-safe: no I/O, no env reads, no side effects at import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from shop_arena.gen.data_synth._synth_helpers import StageSynthError
from shop_arena.gen.data_synth.schema import Navigation, NavigationItem, NavigationItemType
from shop_arena.gen.steps.base import FileInput, InputRef, StepContext

_PHASE: Final[str] = "data_synth"
_STEP_ID: Final[str] = "ingest_navigation"
_STEP_VERSION: Final[int] = 1

_PREFETCH_NAVIGATION: Final[Path] = Path("artifact") / "prefetch" / "navigation.json"
_OUT_NAVIGATION: Final[Path] = Path(".shop_gen") / "stage_cache" / "navigation.json"

_TYPE_BY_PREFIX: Final[tuple[tuple[str, NavigationItemType], ...]] = (
    ("/collections/", "COLLECTION"),
    ("/products/", "PRODUCT"),
    ("/pages/", "PAGE"),
    ("/blogs/", "BLOG"),
)
"""Ordered path-prefix → item-type table. First match wins; anything that
matches no prefix (including off-host links) classifies as ``HTTP``."""


# --------------------------------------------------------------------------- #
# Raw-feed input models (open schema — capture may carry extra fields)
# --------------------------------------------------------------------------- #

_RAW_CONFIG: Final[ConfigDict] = ConfigDict(extra="ignore", frozen=True)


class RawNavItem(BaseModel):
    """One captured menu node from ``artifact/prefetch/navigation.json``.

    Attributes:
        title: Display label scraped from the anchor's text.
        url: Absolute ``href`` as resolved by the browser DOM.
        children: Nested sub-items (one level deep in practice).
    """

    model_config = _RAW_CONFIG

    title: str
    url: str
    children: list[RawNavItem] = []


class RawNavigationFeed(BaseModel):
    """Top-level captured document written by ``shop_arena.explore``.

    Mirrors the self-describing artifact
    :func:`shop_arena.explore.prefetch.capture_navigation` writes: the store
    ``base_url`` it was captured from plus the ``menus`` map (conventionally
    the ``main-menu`` and ``footer`` handles, each a list of captured root
    items). Embedding ``base_url`` lets :func:`build_navigation` relativize
    same-host links without a storefront URL in the ``gen`` config.

    Attributes:
        base_url: Storefront base URL the capture was taken from.
        menus: Captured menus keyed by handle.
    """

    model_config = _RAW_CONFIG

    base_url: str
    menus: dict[str, list[RawNavItem]]


# --------------------------------------------------------------------------- #
# Verbatim mapper (pure — raw capture → Navigation schema)
# --------------------------------------------------------------------------- #


def _relativize(url: str, *, host: str) -> str:
    """Return ``url`` as a host-relative path when it points at ``host``.

    Strips scheme + host (and any query/fragment) for same-host links so the
    stored URL matches the ``/collections/foo`` convention the schema and
    ``shop_backend`` expect. Off-host links are returned unchanged so external
    references stay resolvable.

    Args:
        url: Absolute URL captured from the DOM.
        host: The store's network location (``urlsplit(base_url).netloc``).

    Returns:
        A host-relative path (``/collections/hats``) for same-host links, or
        the original absolute ``url`` for off-host links.
    """
    parts = urlsplit(url)
    if parts.netloc and parts.netloc != host:
        return url
    return parts.path or "/"


def _classify(url: str) -> NavigationItemType:
    """Classify a (relativized) ``url`` into a navigation item type by prefix."""
    for prefix, item_type in _TYPE_BY_PREFIX:
        if url.startswith(prefix):
            return item_type
    return "HTTP"


def _item_from_raw(raw: RawNavItem, *, host: str) -> NavigationItem:
    """Map one raw captured node onto :class:`NavigationItem`, recursing children."""
    url = _relativize(raw.url, host=host)
    return NavigationItem(
        title=raw.title,
        url=url,
        type=_classify(url),
        children=[_item_from_raw(child, host=host) for child in raw.children],
    )


def build_navigation(feed: RawNavigationFeed) -> Navigation:
    """Transform the raw captured menus into a verbatim :class:`Navigation`.

    Each menu's items are relativized against the capture's ``base_url``
    host, classified by path prefix, and recursed into their children. Menu
    handles and item order are preserved exactly as captured.

    Args:
        feed: Parsed ``artifact/prefetch/navigation.json`` document. Its
            ``base_url`` host decides which links are same-host (and thus
            relativized).

    Returns:
        The verbatim :class:`Navigation` payload.
    """
    host = urlsplit(feed.base_url).netloc
    menus = {
        handle: [_item_from_raw(item, host=host) for item in items]
        for handle, items in feed.menus.items()
    }
    return Navigation(menus)


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #


class IngestNavigationStep:
    """Phase 2 ``ingest_navigation`` step — verbatim menus from the capture.

    Reads the seed's ``artifact/prefetch/navigation.json`` (captured by
    ``shop_arena.explore``), maps it into the :class:`Navigation` schema, and
    writes the cached payload under ``.shop_gen/stage_cache/navigation.json``
    — the same path :class:`~shop_arena.gen.data_synth.navigation.SynthNavigationStep`
    writes, so ``assemble_data`` is unaffected. Fully deterministic:
    ``ctx.runtime`` is unused.

    Attributes:
        id: Step id (``ingest_navigation``).
        phase: ``data_synth``.
        inputs: The captured navigation feed as a :class:`FileInput` (empty
            in the ``--list-steps`` placeholder branch).
        outputs: ``.shop_gen/stage_cache/navigation.json``.
        depends_on: Empty — the step depends only on the seed capture file.
        version: Bumped when the ingest behaviour changes (spec §5.7.1).
    """

    def __init__(self, *, seed_dir: Path | None = None) -> None:
        """Build the step bound to a seed directory.

        Args:
            seed_dir: The single seed's ``shop_manuals/<domain>/<run_id>/``
                directory. ``None`` is the ``--list-steps`` placeholder
                branch; the step still surfaces its id with no file inputs.
        """
        self.id: str = _STEP_ID
        self.phase: str = _PHASE
        self.inputs: list[InputRef] = (
            [] if seed_dir is None else [FileInput(path=seed_dir / _PREFETCH_NAVIGATION)]
        )
        self.outputs: list[Path] = [_OUT_NAVIGATION]
        self.depends_on: list[str] = []
        self.version: int = _STEP_VERSION
        self._seed_dir: Path | None = seed_dir

    def run(self, ctx: StepContext) -> None:
        """Ingest the captured navigation into the stage cache.

        Args:
            ctx: Execution context. ``ctx.runtime`` is unused.

        Raises:
            StageSynthError: No seed dir was bound, or the capture file is
                missing, malformed, or fails schema validation.
        """
        if self._seed_dir is None:
            raise StageSynthError(f"{_STEP_ID}: no seed directory bound to the step")

        feed = _load_feed(self._seed_dir / _PREFETCH_NAVIGATION)
        navigation = build_navigation(feed)

        out_path = ctx.out_dir / _OUT_NAVIGATION
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(navigation.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# I/O helper
# --------------------------------------------------------------------------- #


def _load_feed(path: Path) -> RawNavigationFeed:
    """Read + validate the captured navigation feed into :class:`RawNavigationFeed`."""
    if not path.exists():
        raise StageSynthError(
            f"{_STEP_ID}: prefetch navigation.json not found at {path}; "
            "catalog_source='ingest' requires a shop-explore run that captured it "
            "(re-run shop-explore to capture the real navigation)",
        )
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageSynthError(
            f"{_STEP_ID}: navigation.json at {path} is not valid JSON: {exc}",
        ) from exc
    try:
        return RawNavigationFeed.model_validate(raw)
    except ValidationError as exc:
        raise StageSynthError(
            f"{_STEP_ID}: navigation.json at {path} failed RawNavigationFeed validation: {exc}",
        ) from exc


__all__ = [
    "IngestNavigationStep",
    "RawNavItem",
    "RawNavigationFeed",
    "build_navigation",
]
