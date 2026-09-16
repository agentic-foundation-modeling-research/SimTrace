"""``ingest_catalog`` — verbatim catalog ingest from real storefront feeds.

Alternative to the LLM catalog-synthesis chain (``synth_collections`` →
``synth_product_skeletons`` → ``synth_product_details`` → ``synth_alt_text``
→ ``gen_images``). Instead of inventing a fake catalog, this step reads the
real website feeds that ``shop_arena.explore`` captured at
``<seed>/artifact/prefetch/products.json`` and ``collections.json`` and maps
them **verbatim** into the SandboxShop dataset schema
(:mod:`shop_arena.gen.data_synth.schema`): real titles, prices, variants,
options, descriptions, timestamps, and remote CDN image URLs are preserved
byte-for-byte.

Two facts about the real website feeds shape the design:

* The flat feeds carry no product→collection membership. When
  ``shop_arena.explore`` also captured per-collection membership at
  ``<seed>/artifact/prefetch/products_collections.json``, that map is used
  verbatim and the collection set mirrors the real website exactly, save
  for collections that end up with no members (dropped — see
  :func:`build_catalog`) — a product the real website lists in no collection
  stays that way (still reachable by handle and search). Otherwise membership
  is reconstructed by a
  deterministic token-matching heuristic (:func:`derive_membership`), and
  products the heuristic leaves unmatched are swept into a synthesized
  catch-all collection so every product stays reachable.
* Vendor / brand names are kept verbatim (the assemble-time brand scrub is
  disabled — see :mod:`shop_arena.gen.data_synth.assemble`).

The step owns three outputs: ``data/products.json`` and
``data/collections.json`` (final schema) plus
``.shop_gen/stage_cache/collections.json`` in the
:class:`~shop_arena.gen.data_synth.collections.CollectionDraft` shape so the
unchanged ``synth_navigation`` step can still enumerate collection handles.

The step owns three outputs: ``data/products.json`` and ``data/collections.json`` (final schema) plus
``product_type`` and opaque PIM-code tags (the token signal then comes from
titles + handles alone); on such seeds the catch-all collection is expected
to carry a meaningful share of the catalog.

Module is import-safe: no I/O, no env reads, no side effects at import.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from shop_arena.gen.data_synth._synth_helpers import StageSynthError
from shop_arena.gen.data_synth.collections import CollectionDraft
from shop_arena.gen.data_synth.schema import (
    Collection,
    Product,
    ProductImage,
    ProductOption,
    ProductVariant,
)
from shop_arena.gen.steps.base import FileInput, InputRef, StepContext

_PHASE: Final[str] = "data_synth"
_STEP_ID: Final[str] = "ingest_catalog"
_STEP_VERSION: Final[int] = 2

_PREFETCH_DIR: Final[Path] = Path("artifact") / "prefetch"
_PREFETCH_PRODUCTS: Final[Path] = _PREFETCH_DIR / "products.json"
_PREFETCH_COLLECTIONS: Final[Path] = _PREFETCH_DIR / "collections.json"
_PREFETCH_PRODUCTS_COLLECTIONS: Final[Path] = _PREFETCH_DIR / "products_collections.json"

_DATA_DIR: Final[Path] = Path("data")
_OUT_PRODUCTS: Final[Path] = _DATA_DIR / "products.json"
_OUT_COLLECTIONS: Final[Path] = _DATA_DIR / "collections.json"
_OUT_DRAFT_COLLECTIONS: Final[Path] = Path(".shop_gen") / "stage_cache" / "collections.json"

_CATCH_ALL_TITLE: Final[str] = "All Products"
_CATCH_ALL_HANDLE: Final[str] = "all"
_CATCH_ALL_SORT_ORDER: Final[str] = "manual"

_ID_HEX_CHARS: Final[int] = 12
"""Leading hex digits of ``sha256(scope + key)`` used for the catch-all id.

Mirrors :data:`shop_arena.gen.data_synth.assemble._ID_HEX_CHARS`; only the
synthesized catch-all collection needs a hashed id — every other record
reuses its verbatim Shopify id.
"""

_MIN_TOKEN_LEN: Final[int] = 3
"""Membership tokens shorter than this are dropped as noise."""

_GENERIC_DF_FRACTION: Final[float] = 0.4
"""A token appearing in more than this fraction of collections is treated as
generic and dropped from every collection's distinctive-token set, so common
words (e.g. ``women``/``men``/``de``) cannot drive matches on their own."""

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
"""Lowercase alphanumeric run — the membership tokenizer."""

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # English function words
        "the", "and", "for", "with", "from", "all", "new", "our", "your",
        "this", "that", "you", "are", "has",
        # French function words (real feeds are frequently non-English)
        "des", "les", "une", "uns", "aux", "par", "pour", "avec", "sur",
        "sans", "vos", "nos", "ami",
    },
)
"""Small closed set of high-frequency function words dropped before matching."""


# --------------------------------------------------------------------------- #
# Raw-feed input models (open schema — Shopify ships many extra fields)
# --------------------------------------------------------------------------- #

_RAW_CONFIG: Final[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
"""Ignore the many Shopify fields we do not carry (``grams``, ``taxable``,
``variant_ids``, ``featured_image``, ``product_id``, ``products_count``, ...)."""


class RawOption(BaseModel):
    """One product option group in the raw ``/products.json`` feed."""

    model_config = _RAW_CONFIG

    name: str
    position: int
    values: list[str]


class RawVariant(BaseModel):
    """One purchasable variant in the raw ``/products.json`` feed."""

    model_config = _RAW_CONFIG

    id: int
    title: str
    sku: str | None = None
    price: str
    compare_at_price: str | None = None
    available: bool
    option1: str | None = None
    option2: str | None = None
    option3: str | None = None
    position: int
    requires_shipping: bool


class RawImage(BaseModel):
    """One image in the raw feed (product image or collection hero).

    ``id`` / ``position`` / ``width`` / ``height`` are optional because
    collection hero images in ``/collections.json`` omit them; product
    images always carry all four.
    """

    model_config = _RAW_CONFIG

    id: int | None = None
    src: str
    alt: str | None = None
    width: int | None = None
    height: int | None = None
    position: int | None = None


class RawProduct(BaseModel):
    """One product record in the raw ``/products.json`` feed."""

    model_config = _RAW_CONFIG

    id: int
    title: str
    handle: str
    body_html: str | None = None
    published_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    vendor: str
    product_type: str
    tags: list[str]
    variants: list[RawVariant]
    images: list[RawImage]
    options: list[RawOption]


class RawCollection(BaseModel):
    """One collection record in the raw ``/collections.json`` feed."""

    model_config = _RAW_CONFIG

    id: int
    title: str
    handle: str
    description: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    image: RawImage | None = None


class RawProductsFeed(BaseModel):
    """Top-level ``{"products": [...]}`` document."""

    model_config = _RAW_CONFIG

    products: list[RawProduct]


class RawCollectionsFeed(BaseModel):
    """Top-level ``{"collections": [...]}`` document."""

    model_config = _RAW_CONFIG

    collections: list[RawCollection]


class RawProductsCollectionsFeed(RootModel[dict[str, list[str]]]):
    """Ground-truth membership captured by ``shop_arena.explore``.

    The ``products_collections.json`` prefetch artifact is a flat object
    mapping each collection handle to its member product handles (fetched
    from ``/collections/<handle>/products.json``). ``root`` yields that
    mapping directly for :func:`build_catalog`.
    """


# --------------------------------------------------------------------------- #
# Verbatim mappers (pure — raw feed record → final schema record)
# --------------------------------------------------------------------------- #


def variant_from_raw(raw: RawVariant) -> ProductVariant:
    """Map a raw feed variant onto :class:`ProductVariant` verbatim."""
    return ProductVariant(
        id=raw.id,
        title=raw.title,
        sku=raw.sku,
        price=raw.price,
        compare_at_price=raw.compare_at_price,
        available=raw.available,
        option1=raw.option1,
        option2=raw.option2,
        option3=raw.option3,
        position=raw.position,
        requires_shipping=raw.requires_shipping,
    )


def image_from_raw(raw: RawImage, *, fallback_id: int, fallback_position: int) -> ProductImage:
    """Map a raw feed image onto :class:`ProductImage`, keeping the remote ``src``.

    Args:
        raw: Raw image record. Its absolute CDN ``src`` is preserved
            verbatim; ``shop_backend`` passes absolute URLs through
            unchanged.
        fallback_id: Numeric id used when the raw image omits ``id`` (e.g.
            a collection hero image).
        fallback_position: 1-indexed position used when the raw image omits
            ``position``.
    """
    return ProductImage(
        id=raw.id if raw.id is not None else fallback_id,
        src=raw.src,
        alt=raw.alt,
        width=raw.width,
        height=raw.height,
        position=raw.position if raw.position is not None else fallback_position,
    )


def product_from_raw(raw: RawProduct) -> Product:
    """Map a raw feed product onto :class:`Product` verbatim.

    ``body_html`` becomes ``description_html`` (an absent body maps to the
    empty string); vendor, tags, timestamps, options, variants, and images
    are carried through as-is. Image ``alt`` defaults to the feed value
    (usually ``None``).
    """
    return Product(
        id=raw.id,
        title=raw.title,
        handle=raw.handle,
        description_html=raw.body_html or "",
        vendor=raw.vendor,
        product_type=raw.product_type,
        tags=list(raw.tags),
        published_at=raw.published_at or "",
        created_at=raw.created_at or "",
        updated_at=raw.updated_at or "",
        options=[
            ProductOption(name=opt.name, position=opt.position, values=list(opt.values))
            for opt in raw.options
        ],
        variants=[variant_from_raw(v) for v in raw.variants],
        images=[
            image_from_raw(img, fallback_id=raw.id, fallback_position=position)
            for position, img in enumerate(raw.images, start=1)
        ],
    )


def collection_from_raw(raw: RawCollection, *, product_handles: list[str]) -> Collection:
    """Map a raw feed collection onto :class:`Collection` verbatim.

    ``description`` carries HTML in the feed, so it becomes
    ``description_html``; the plain-text ``description`` field is left
    ``None`` (the feed ships no plain-text variant). Membership is supplied
    by the caller via ``product_handles``.
    """
    image = (
        image_from_raw(raw.image, fallback_id=raw.id, fallback_position=1)
        if raw.image is not None
        else None
    )
    return Collection(
        id=raw.id,
        title=raw.title,
        handle=raw.handle,
        description=None,
        description_html=raw.description,
        image=image,
        published_at=raw.published_at,
        updated_at=raw.updated_at,
        sort_order=None,
        product_handles=product_handles,
    )


# --------------------------------------------------------------------------- #
# Membership heuristic
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> set[str]:
    """Return the distinctive lowercase tokens of ``text``."""
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LEN and token not in _STOPWORDS
    }


def _product_tokens(raw: RawProduct) -> set[str]:
    """Union of tokens drawn from a product's title, handle, tags, and type."""
    tokens = _tokenize(raw.title) | _tokenize(raw.handle) | _tokenize(raw.product_type)
    for tag in raw.tags:
        tokens |= _tokenize(tag)
    return tokens


def derive_membership(
    products: list[RawProduct],
    collections: list[RawCollection],
) -> dict[str, list[str]]:
    """Reconstruct product→collection membership by token matching.

    The feeds carry no membership, so this deterministic heuristic assigns
    each product to every collection sharing at least one *distinctive*
    token. A collection's distinctive tokens are its title + handle tokens
    minus any token appearing in more than :data:`_GENERIC_DF_FRACTION` of
    all collections (so generic words cannot drive matches). A token unique
    to a single collection is never suppressed — it cannot be generic — which
    also keeps the heuristic well-behaved for small catalogs. Products may
    join multiple collections (Shopify semantics). Output order is fully
    determined by feed order.

    Args:
        products: Raw product records, in feed order.
        collections: Raw collection records, in feed order.

    Returns:
        Mapping of ``collection.handle`` to the list of member product
        handles, in feed order.I Every collection handle is present (with a
        possibly-empty list). Products matching no collection are absent
        from every list — the caller sweeps them into a catch-all.
    """
    collection_tokens: dict[str, set[str]] = {
        c.handle: _tokenize(c.title) | _tokenize(c.handle) for c in collections
    }
    document_frequency: Counter[str] = Counter()
    for tokens in collection_tokens.values():
        document_frequency.update(tokens)
    max_frequency = _GENERIC_DF_FRACTION * len(collections)
    distinctive: dict[str, set[str]] = {
        handle: {
            t
            for t in tokens
            if document_frequency[t] == 1 or document_frequency[t] <= max_frequency
        }
        for handle, tokens in collection_tokens.items()
    }

    membership: dict[str, list[str]] = {c.handle: [] for c in collections}
    for product in products:
        product_tokens = _product_tokens(product)
        for collection in collections:
            if distinctive[collection.handle] & product_tokens:
                membership[collection.handle].append(product.handle)
    return membership


# --------------------------------------------------------------------------- #
# Catalog assembly (pure)
# --------------------------------------------------------------------------- #


class IngestedCatalog:
    """Bundle of the verbatim products and collections ready to serialise.

    Attributes:
        products: Verbatim :class:`Product` list for ``data/products.json``.
        collections: Verbatim :class:`Collection` list (including any
            catch-all) for ``data/collections.json``.
    """

    __slots__ = ("collections", "products")

    def __init__(self, *, products: list[Product], collections: list[Collection]) -> None:
        self.products = products
        self.collections = collections


def build_catalog(
    products_feed: RawProductsFeed,
    collections_feed: RawCollectionsFeed,
    *,
    membership: dict[str, list[str]] | None = None,
) -> IngestedCatalog:
    """Transform the raw feeds into verbatim products + collections.

    Attaches ``product_handles`` to each collection and drops any collection
    that ends up with no members (an empty collection resolves by handle but
    returns zero products, breaking the hosting check). When membership is
    reconstructed heuristically (``membership is None``), any product left
    unmatched is swept into a synthesized catch-all collection so every product
    stays reachable. When ground-truth ``membership`` is supplied, the
    collection set mirrors the real storefront exactly (minus empties) and no
    catch-all is added (an unmatched product stays reachable by handle and
    search).

    Args:
        products_feed: Parsed ``/products.json`` document.
        collections_feed: Parsed ``/collections.json`` document.
        membership: Ground-truth ``collection.handle`` → member product
            handles captured by ``shop_arena.explore`` (see
            :class:`RawProductsCollectionsFeed`). When ``None`` (no
            ``products_collections.json`` in the seed), membership is
            reconstructed heuristically via :func:`derive_membership`. A
            supplied map is used verbatim, restricted to product handles that
            actually exist in ``products_feed`` (a collection absent from the
            map contributes no members).

    Returns:
        The assembled :class:`IngestedCatalog`.
    """
    raw_products = products_feed.products
    raw_collections = collections_feed.collections

    products = [product_from_raw(p) for p in raw_products]

    if membership is None:
        resolved = derive_membership(raw_products, raw_collections)
    else:
        known = {p.handle for p in raw_products}
        resolved = {
            c.handle: [h for h in membership.get(c.handle, []) if h in known]
            for c in raw_collections
        }
    # Drop collections with no members: an empty collection resolves by handle
    # but returns zero products, which breaks the hosting check and is dead
    # weight in the storefront (promotional / smart collections such as
    # "buy 2 get 1" carry no static membership in the flat feed). Products in a
    # dropped collection stay reachable by handle and search, and (in the
    # heuristic path) are swept into the catch-all below.
    collections = [
        collection_from_raw(c, product_handles=resolved[c.handle])
        for c in raw_collections
        if resolved[c.handle]
    ]

    # The weak token heuristic leaves many products unmatched, so sweep them
    # into a synthesized catch-all to keep every product reachable. Ground-truth
    # membership from ``shop_arena.explore`` is authoritative and mirrors the
    # real storefront's collection set exactly, so no catch-all is added there:
    # a product the real store lists in no collection stays that way (still
    # reachable by handle and search), rather than inventing an extra collection.
    if membership is None:
        matched = {handle for handles in resolved.values() for handle in handles}
        unmatched = [p.handle for p in raw_products if p.handle not in matched]
        if unmatched:
            collections.append(
                _catch_all_collection(unmatched, taken={c.handle for c in collections}),
            )

    return IngestedCatalog(products=products, collections=collections)


def _catch_all_collection(product_handles: list[str], *, taken: set[str]) -> Collection:
    """Build the synthesized catch-all collection for unmatched products."""
    handle = _CATCH_ALL_HANDLE
    suffix = 2
    while handle in taken:
        handle = f"{_CATCH_ALL_HANDLE}-{suffix}"
        suffix += 1
    return Collection(
        id=_numeric_id("collection", handle),
        title=_CATCH_ALL_TITLE,
        handle=handle,
        description=None,
        description_html=None,
        image=None,
        published_at=None,
        updated_at=None,
        sort_order=_CATCH_ALL_SORT_ORDER,
        product_handles=product_handles,
    )


def _draft_from_collection(collection: Collection) -> CollectionDraft:
    """Project a :class:`Collection` onto the navigation-facing draft shape.

    ``synth_navigation`` reads the cached draft list only to enumerate
    collection handles; ``description`` is prompt context, so the title is a
    safe non-empty stand-in for the HTML-only feed description.
    """
    return CollectionDraft(
        title=collection.title,
        handle=collection.handle,
        description=collection.title,
        sort_order=collection.sort_order or _CATCH_ALL_SORT_ORDER,
        target_product_count=max(1, len(collection.product_handles)),
    )


def _numeric_id(scope: str, key: str) -> int:
    """Deterministic non-negative id for ``(scope, key)`` (see assemble.py)."""
    digest = hashlib.sha256(f"{scope}\0{key}".encode()).hexdigest()
    return int(digest[:_ID_HEX_CHARS], 16)


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #


class IngestCatalogStep:
    """Phase 2 ``ingest_catalog`` step — verbatim catalog from prefetch feeds.

    Reads the seed's ``artifact/prefetch/{products,collections}.json`` (plus
    the optional ``products_collections.json`` membership map), transforms them
    into the final dataset schema, resolves membership, and writes
    ``data/products.json``, ``data/collections.json``, and the
    navigation-facing draft cache. Fully deterministic — ``ctx.runtime`` is
    unused.

    Attributes:
        id: Step id (``ingest_catalog``).
        phase: ``data_synth``.
        inputs: The two required prefetch feeds as :class:`FileInput`\\s, plus
            ``products_collections.json`` when the seed captured it (empty in
            the ``--list-steps`` placeholder branch).
        outputs: ``data/products.json``, ``data/collections.json``, and
            ``.shop_gen/stage_cache/collections.json``.
        depends_on: Empty — the step depends only on the seed feed files.
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
        self.inputs: list[InputRef] = self._build_inputs(seed_dir)
        self.outputs: list[Path] = [_OUT_PRODUCTS, _OUT_COLLECTIONS, _OUT_DRAFT_COLLECTIONS]
        self.depends_on: list[str] = []
        self.version: int = _STEP_VERSION
        self._seed_dir: Path | None = seed_dir

    @staticmethod
    def _build_inputs(seed_dir: Path | None) -> list[InputRef]:
        """Return the step's file inputs for ``seed_dir``.

        The two flat feeds are always declared. ``products_collections.json``
        is declared only when it exists on disk: a declared
        :class:`FileInput` pointing at a missing file makes
        :func:`~shop_arena.gen.steps.state.compute_fingerprint` raise, so
        seeds captured before membership prefetch existed must not reference
        it.
        """
        if seed_dir is None:
            return []
        inputs: list[InputRef] = [
            FileInput(path=seed_dir / _PREFETCH_PRODUCTS),
            FileInput(path=seed_dir / _PREFETCH_COLLECTIONS),
        ]
        if (seed_dir / _PREFETCH_PRODUCTS_COLLECTIONS).exists():
            inputs.append(FileInput(path=seed_dir / _PREFETCH_PRODUCTS_COLLECTIONS))
        return inputs

    def run(self, ctx: StepContext) -> None:
        """Ingest the prefetch feeds into the final catalog files.

        Args:
            ctx: Execution context. ``ctx.runtime`` is unused.

        Raises:
            StageSynthError: A seed dir was not bound, or a feed is missing,
                malformed, or fails schema validation.
        """
        if self._seed_dir is None:
            raise StageSynthError(f"{_STEP_ID}: no seed directory bound to the step")

        products_feed = _load_feed(
            self._seed_dir / _PREFETCH_PRODUCTS,
            model=RawProductsFeed,
            label="products.json",
        )
        collections_feed = _load_feed(
            self._seed_dir / _PREFETCH_COLLECTIONS,
            model=RawCollectionsFeed,
            label="collections.json",
        )

        membership_path = self._seed_dir / _PREFETCH_PRODUCTS_COLLECTIONS
        membership = (
            _load_feed(
                membership_path,
                model=RawProductsCollectionsFeed,
                label="products_collections.json",
            ).root
            if membership_path.exists()
            else None
        )

        catalog = build_catalog(products_feed, collections_feed, membership=membership)
        drafts = [_draft_from_collection(c) for c in catalog.collections]

        _write_json(
            ctx.out_dir / _OUT_PRODUCTS,
            [p.model_dump(mode="json") for p in catalog.products],
        )
        _write_json(
            ctx.out_dir / _OUT_COLLECTIONS,
            [c.model_dump(mode="json") for c in catalog.collections],
        )
        _write_json(
            ctx.out_dir / _OUT_DRAFT_COLLECTIONS,
            [d.model_dump(mode="json") for d in drafts],
        )


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def _load_feed[T: BaseModel](path: Path, *, model: type[T], label: str) -> T:
    """Read + validate a prefetch feed file into ``model``."""
    if not path.exists():
        raise StageSynthError(
            f"{_STEP_ID}: prefetch {label} not found at {path}; "
            "catalog_source='ingest' requires a shop-explore run that captured it",
        )
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageSynthError(f"{_STEP_ID}: {label} at {path} is not valid JSON: {exc}") from exc
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise StageSynthError(
            f"{_STEP_ID}: {label} at {path} failed {model.__name__} validation: {exc}",
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    """Serialise ``payload`` with the pipeline's canonical JSON convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "IngestCatalogStep",
    "IngestedCatalog",
    "RawCollection",
    "RawCollectionsFeed",
    "RawProduct",
    "RawProductsCollectionsFeed",
    "RawProductsFeed",
    "build_catalog",
    "collection_from_raw",
    "derive_membership",
    "image_from_raw",
    "product_from_raw",
    "variant_from_raw",
]
