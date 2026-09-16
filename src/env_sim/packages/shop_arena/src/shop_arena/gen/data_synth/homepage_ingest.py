"""``ingest_homepage`` — verbatim site-level hero / banner ingest.

Turns the storefront's real editorial imagery — captured with a headless
browser by :func:`shop_arena.explore.prefetch.capture_hero` into
``<seed>/artifact/prefetch/hero.json`` — into the SandboxShop
``data/homepage.json`` file so the twin's homepage reuses the source site's
own hero and promo banners instead of improvising them.

Unlike the catalog/navigation ingest steps, this one runs in **both** catalog
modes (synth *and* ingest): the hero/banner imagery is site-level and
independent of whether the catalog was synthesized or copied. It is also
**best-effort** end-to-end — the hero capture itself is optional (Playwright
may be unavailable), so a missing or malformed ``hero.json`` is not an error:
the step writes an empty homepage (``hero=null, banners=[]``) and returns. The
storefront then falls back to its image-less layout.

The captured URLs are hotlinked verbatim: the absolute CDN ``src`` is stored
as-is (``shop_backend`` passes absolute URLs through untouched) and **no**
bytes are downloaded. Real ``alt`` text is deliberately dropped
(``alt=None``) to avoid leaking brand copy into the twin.

The step owns ``data/homepage.json`` outright (no ``assemble_data``
involvement), mirroring how ``ingest_catalog`` owns its own ``data/*.json``
files.

Module is import-safe: no I/O, no env reads, no side effects at import.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from shop_arena.gen.data_synth.schema import Homepage, ProductImage
from shop_arena.gen.steps.base import FileInput, InputRef, StepContext

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_PHASE: Final[str] = "data_synth"
_STEP_ID: Final[str] = "ingest_homepage"
_STEP_VERSION: Final[int] = 1

_PREFETCH_HERO: Final[Path] = Path("artifact") / "prefetch" / "hero.json"
_OUT_HOMEPAGE: Final[Path] = Path("data") / "homepage.json"

_ID_HEX_CHARS: Final[int] = 12
"""Leading hex digits of ``sha256(scope + url)`` used for the image id.

Mirrors :data:`shop_arena.gen.data_synth.ingest._ID_HEX_CHARS`."""

_HERO_SCOPE: Final[str] = "homepage-hero"
_BANNER_SCOPE: Final[str] = "homepage-banner"


# --------------------------------------------------------------------------- #
# Raw-feed input models (open schema — capture may carry extra fields)
# --------------------------------------------------------------------------- #

_RAW_CONFIG: Final[ConfigDict] = ConfigDict(extra="ignore", frozen=True)


class RawHeroImage(BaseModel):
    """One captured image from ``artifact/prefetch/hero.json``.

    Attributes:
        url: Absolute image URL as resolved by the browser DOM.
        width: Rendered/natural pixel width, or ``None`` when unknown.
        height: Rendered/natural pixel height, or ``None`` when unknown.
        alt: Scraped alt text (dropped on mapping for brand safety).
    """

    model_config = _RAW_CONFIG

    url: str
    width: int | None = None
    height: int | None = None
    alt: str | None = None


class RawHeroFeed(BaseModel):
    """Top-level captured document written by ``shop_arena.explore``.

    Mirrors the artifact :func:`shop_arena.explore.prefetch.capture_hero`
    writes: the store ``base_url``, the single ``hero`` (or ``None``), and the
    ordered ``banners`` list.

    Attributes:
        base_url: Storefront base URL the capture was taken from.
        hero: The captured hero image, or ``None`` when none cleared the
            capture thresholds.
        banners: Captured promo banners in scroll order (possibly empty).
    """

    model_config = _RAW_CONFIG

    base_url: str = ""
    hero: RawHeroImage | None = None
    banners: list[RawHeroImage] = []


# --------------------------------------------------------------------------- #
# Verbatim mapper (pure — raw capture → Homepage schema)
# --------------------------------------------------------------------------- #


def _numeric_id(scope: str, key: str) -> int:
    """Deterministic non-negative id for ``(scope, key)`` (see ingest.py)."""
    digest = hashlib.sha256(f"{scope}\0{key}".encode()).hexdigest()
    return int(digest[:_ID_HEX_CHARS], 16)


def _image_from_raw(raw: RawHeroImage, *, scope: str, position: int) -> ProductImage:
    """Map one captured image onto :class:`ProductImage`, hotlinking its URL.

    The absolute CDN ``url`` becomes ``src`` verbatim; ``alt`` is dropped to
    keep real brand copy out of the twin; missing dimensions default to ``0``
    (advisory metadata — the storefront ``<img>`` renders at natural size).

    Args:
        raw: Captured image record.
        scope: Id namespace distinguishing the hero from banners.
        position: 1-indexed display position within the homepage.
    """
    return ProductImage(
        id=_numeric_id(scope, raw.url),
        src=raw.url,
        alt=None,
        width=raw.width or 0,
        height=raw.height or 0,
        position=position,
    )


def build_homepage(feed: RawHeroFeed) -> Homepage:
    """Transform the raw captured hero/banners into a :class:`Homepage`.

    Args:
        feed: Parsed ``artifact/prefetch/hero.json`` document.

    Returns:
        The :class:`Homepage` payload — ``hero`` is ``None`` when the capture
        found none; ``banners`` preserves capture order.
    """
    hero = None if feed.hero is None else _image_from_raw(feed.hero, scope=_HERO_SCOPE, position=1)
    banners = [
        _image_from_raw(banner, scope=_BANNER_SCOPE, position=position)
        for position, banner in enumerate(feed.banners, start=1)
    ]
    return Homepage(hero=hero, banners=banners)


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #


class IngestHomepageStep:
    """Phase 2 ``ingest_homepage`` step — verbatim hero/banners from the capture.

    Reads the seed's ``artifact/prefetch/hero.json`` (captured by
    ``shop_arena.explore``), maps it into the :class:`Homepage` schema, and
    writes ``data/homepage.json``. Registered in both catalog modes and fully
    deterministic (``ctx.runtime`` is unused).

    Best-effort: a missing seed dir, a missing capture file, or a malformed
    one all resolve to an empty homepage rather than an error, because the
    upstream hero capture is itself optional.

    Attributes:
        id: Step id (``ingest_homepage``).
        phase: ``data_synth``.
        inputs: The captured hero feed as a :class:`FileInput`, declared only
            when ``hero.json`` exists on disk (empty when no seed is bound or
            the best-effort capture did not run).
        outputs: ``data/homepage.json``.
        depends_on: Empty — the step depends only on the seed capture file.
        version: Bumped when the ingest behaviour changes (spec §5.7.1).
    """

    def __init__(self, *, seed_dir: Path | None = None) -> None:
        """Build the step bound to a seed directory.

        Args:
            seed_dir: The single seed's ``shop_manuals/<domain>/<run_id>/``
                directory. ``None`` is the ``--list-steps`` placeholder branch
                or a multi-seed run; the step still surfaces its id and writes
                an empty homepage at run time.
        """
        self.id: str = _STEP_ID
        self.phase: str = _PHASE
        self.inputs: list[InputRef] = self._build_inputs(seed_dir)
        self.outputs: list[Path] = [_OUT_HOMEPAGE]
        self.depends_on: list[str] = []
        self.version: int = _STEP_VERSION
        self._seed_dir: Path | None = seed_dir

    @staticmethod
    def _build_inputs(seed_dir: Path | None) -> list[InputRef]:
        """Return the step's file inputs for ``seed_dir``.

        The captured ``hero.json`` is declared as a :class:`FileInput` only
        when it exists on disk. The hero capture is best-effort (Playwright may
        be unavailable), so the file is frequently absent — and a declared
        :class:`FileInput` pointing at a missing file makes
        :func:`~shop_arena.gen.steps.state.compute_fingerprint` raise. When it
        is absent the step still runs, writing an empty homepage; staleness
        then rests on the step version alone.
        """
        if seed_dir is None:
            return []
        hero_path = seed_dir / _PREFETCH_HERO
        return [FileInput(path=hero_path)] if hero_path.exists() else []

    def run(self, ctx: StepContext) -> None:
        """Ingest the captured hero/banners into ``data/homepage.json``.

        Args:
            ctx: Execution context. ``ctx.runtime`` is unused.
        """
        feed = self._load_feed()
        homepage = build_homepage(feed)

        out_path = ctx.out_dir / _OUT_HOMEPAGE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(homepage.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _LOGGER.info(
            "ingest_homepage: hero=%s banners=%d",
            "yes" if homepage.hero is not None else "no",
            len(homepage.banners),
        )

    def _load_feed(self) -> RawHeroFeed:
        """Load the captured hero feed, defaulting to empty on any problem.

        Returns:
            The parsed :class:`RawHeroFeed`, or an empty one when no seed is
            bound, the capture file is absent, or it fails to parse/validate.
        """
        if self._seed_dir is None:
            return RawHeroFeed()
        path = self._seed_dir / _PREFETCH_HERO
        if not path.exists():
            _LOGGER.info("ingest_homepage: no hero.json at %s; writing empty homepage", path)
            return RawHeroFeed()
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            return RawHeroFeed.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            _LOGGER.warning(
                "ingest_homepage: hero.json at %s is unusable (%s); writing empty homepage",
                path,
                exc,
            )
            return RawHeroFeed()


__all__ = [
    "IngestHomepageStep",
    "RawHeroFeed",
    "RawHeroImage",
    "build_homepage",
]
