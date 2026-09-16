"""Unit tests for :mod:`shop_arena.gen.data_synth.ingest`.

Covers the verbatim catalog-ingest mode:

* Field-verbatim mapping: raw feed variants / images / products /
  collections map onto the closed schema with extras dropped, absolute
  CDN ``src`` preserved, and Shopify ids reused; the results validate.
* :func:`derive_membership`: distinctive-token matching, high-document-
  frequency token suppression, multi-membership, and determinism.
* :func:`build_catalog`: catch-all creation for heuristic unmatched
  products, its omission when every product matches, and its omission
  under ground-truth membership (which mirrors the store exactly).
* :class:`IngestCatalogStep`: the step contract, the ``--list-steps``
  placeholder branch, and an end-to-end ``run`` that writes the three
  files with valid, membership-complete contents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shop_arena.gen.config import ShopGenConfig
from shop_arena.gen.data_synth import (
    Collection,
    IngestCatalogStep,
    Product,
    StageSynthError,
    build_catalog,
    derive_membership,
    image_from_raw,
    product_from_raw,
    variant_from_raw,
)
from shop_arena.gen.data_synth.ingest import (
    RawCollection,
    RawCollectionsFeed,
    RawImage,
    RawProduct,
    RawProductsCollectionsFeed,
    RawProductsFeed,
    RawVariant,
)
from shop_arena.gen.steps.base import FileInput, StepContext

# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _raw_product_dict(
    *,
    raw_id: int = 111,
    title: str = "Summer Dresses",
    handle: str = "summer-dresses",
    tags: list[str] | None = None,
    product_type: str = "",
) -> dict[str, object]:
    """A raw ``/products.json`` record with Shopify extras to be dropped."""
    return {
        "id": raw_id,
        "title": title,
        "handle": handle,
        "body_html": "<p>A lovely dress.</p>",
        "published_at": "2023-05-01T10:00:00-04:00",
        "created_at": "2023-04-01T10:00:00-04:00",
        "updated_at": "2023-06-01T10:00:00-04:00",
        "vendor": "Ami Paris",
        "product_type": product_type,
        "tags": tags if tags is not None else ["SS23", "PIM-0042"],
        "variants": [
            {
                "id": 5001,
                "title": "S",
                "sku": "DRS-S",
                "price": "390.00",
                "compare_at_price": None,
                "available": True,
                "option1": "S",
                "option2": None,
                "option3": None,
                "position": 1,
                "requires_shipping": True,
                # Shopify extras that must be ignored:
                "grams": 200,
                "taxable": True,
                "product_id": raw_id,
                "featured_image": None,
                "created_at": "2023-04-01T10:00:00-04:00",
            },
        ],
        "images": [
            {
                "id": 9001,
                "src": "https://cdn.shopify.com/s/files/1/dress-front.jpg",
                "width": 1200,
                "height": 1600,
                "position": 1,
            },
        ],
        "options": [{"name": "Size", "position": 1, "values": ["S", "M"]}],
    }


def _raw_collection_dict(
    *,
    raw_id: int = 222,
    title: str = "Dresses",
    handle: str = "dresses",
) -> dict[str, object]:
    """A raw ``/collections.json`` record."""
    return {
        "id": raw_id,
        "title": title,
        "handle": handle,
        "description": "<p>All our dresses.</p>",
        "published_at": "2023-01-01T00:00:00-05:00",
        "updated_at": "2023-06-01T00:00:00-04:00",
        "image": None,
        # Shopify extra that must be ignored:
        "products_count": 42,
    }


def _raw_product_model(
    *,
    raw_id: int,
    title: str,
    handle: str,
    tags: list[str] | None = None,
    product_type: str = "",
) -> RawProduct:
    """Minimal :class:`RawProduct` for membership tests."""
    return RawProduct(
        id=raw_id,
        title=title,
        handle=handle,
        vendor="Ami Paris",
        product_type=product_type,
        tags=tags if tags is not None else [],
        variants=[],
        images=[],
        options=[],
    )


def _raw_collection_model(*, raw_id: int, title: str, handle: str) -> RawCollection:
    """Minimal :class:`RawCollection` for membership tests."""
    return RawCollection(id=raw_id, title=title, handle=handle)


# --------------------------------------------------------------------------- #
# Verbatim mappers
# --------------------------------------------------------------------------- #


def test_variant_from_raw_copies_fields_verbatim() -> None:
    raw = RawVariant(
        id=5001,
        title="S",
        sku="DRS-S",
        price="390.00",
        compare_at_price="450.00",
        available=True,
        option1="S",
        option2=None,
        option3=None,
        position=1,
        requires_shipping=True,
    )
    variant = variant_from_raw(raw)
    assert variant.id == 5001
    assert variant.price == "390.00"
    assert variant.compare_at_price == "450.00"
    assert variant.sku == "DRS-S"
    assert variant.option1 == "S"
    assert variant.requires_shipping is True


def test_image_from_raw_preserves_absolute_cdn_src() -> None:
    raw = RawImage(
        id=9001,
        src="https://cdn.shopify.com/s/files/1/dress.jpg",
        width=1200,
        height=1600,
        position=2,
    )
    image = image_from_raw(raw, fallback_id=1, fallback_position=1)
    assert image.src == "https://cdn.shopify.com/s/files/1/dress.jpg"
    assert image.id == 9001
    assert image.position == 2
    assert image.alt is None


def test_image_from_raw_uses_fallbacks_when_id_and_position_absent() -> None:
    raw = RawImage(src="https://cdn.example/hero.jpg", width=800, height=600)
    image = image_from_raw(raw, fallback_id=42, fallback_position=7)
    assert image.id == 42
    assert image.position == 7


def test_image_from_raw_allows_missing_width_and_height() -> None:
    """Collection hero images in ``/collections.json`` omit dimensions."""
    raw = RawImage(src="https://cdn.example/hero.jpg")
    image = image_from_raw(raw, fallback_id=42, fallback_position=1)
    assert image.width is None
    assert image.height is None


def test_raw_collections_feed_parses_hero_image_without_dimensions() -> None:
    """A collection hero image lacking ``width``/``height`` validates."""
    feed = RawCollectionsFeed.model_validate(
        {
            "collections": [
                {
                    "id": 1618697224237,
                    "title": "Fire Pits",
                    "handle": "fire-pits",
                    "image": {
                        "id": 1618697224237,
                        "src": "https://cdn.shopify.com/s/files/1/hero.jpg",
                        "alt": "A burning fire pit.",
                    },
                }
            ]
        }
    )
    hero = feed.collections[0].image
    assert hero is not None
    assert hero.width is None
    assert hero.height is None


def test_product_from_raw_maps_verbatim_and_drops_extras() -> None:
    raw = RawProduct.model_validate(_raw_product_dict())
    product = product_from_raw(raw)
    assert isinstance(product, Product)
    assert product.id == 111
    assert product.title == "Summer Dresses"
    assert product.vendor == "Ami Paris"
    assert product.description_html == "<p>A lovely dress.</p>"
    assert product.published_at == "2023-05-01T10:00:00-04:00"
    assert product.tags == ["SS23", "PIM-0042"]
    assert [v.id for v in product.variants] == [5001]
    assert product.variants[0].price == "390.00"
    assert product.images[0].src == "https://cdn.shopify.com/s/files/1/dress-front.jpg"
    assert product.options[0].values == ["S", "M"]


def test_product_from_raw_absent_body_html_maps_to_empty_string() -> None:
    payload = _raw_product_dict()
    del payload["body_html"]
    product = product_from_raw(RawProduct.model_validate(payload))
    assert product.description_html == ""


def test_collection_from_raw_maps_description_html_and_null_plain_text() -> None:
    raw = RawCollection.model_validate(_raw_collection_dict())
    # Give the collection a member so it survives the empty-collection drop;
    # this test only exercises field mapping, not membership.
    product = _raw_product_dict(raw_id=1, title="Summer Dress", handle="summer-dress")
    collection = build_catalog(
        RawProductsFeed.model_validate({"products": [product]}),
        RawCollectionsFeed(collections=[raw]),
        membership={raw.handle: ["summer-dress"]},
    ).collections[0]
    assert isinstance(collection, Collection)
    assert collection.id == 222
    assert collection.handle == "dresses"
    assert collection.description_html == "<p>All our dresses.</p>"
    assert collection.description is None
    assert collection.updated_at == "2023-06-01T00:00:00-04:00"
    assert collection.image is None


# --------------------------------------------------------------------------- #
# derive_membership
# --------------------------------------------------------------------------- #


def test_derive_membership_matches_on_distinctive_token() -> None:
    products = [_raw_product_model(raw_id=1, title="Summer Dresses", handle="summer-dresses")]
    collections = [
        _raw_collection_model(raw_id=10, title="Dresses", handle="dresses"),
        _raw_collection_model(raw_id=11, title="Shoes", handle="shoes"),
    ]
    membership = derive_membership(products, collections)
    assert membership["dresses"] == ["summer-dresses"]
    assert membership["shoes"] == []


def test_derive_membership_suppresses_high_document_frequency_token() -> None:
    """A token in >40% of collections cannot drive a match on its own."""
    collections = [
        _raw_collection_model(raw_id=1, title="Women Dresses", handle="women-dresses"),
        _raw_collection_model(raw_id=2, title="Women Shoes", handle="women-shoes"),
        _raw_collection_model(raw_id=3, title="Women Bags", handle="women-bags"),
        _raw_collection_model(raw_id=4, title="Men Hats", handle="men-hats"),
        _raw_collection_model(raw_id=5, title="Kids Toys", handle="kids-toys"),
    ]
    # "women" appears in 3/5 = 60% of collections → suppressed everywhere.
    products = [_raw_product_model(raw_id=1, title="Women Dresses Summer", handle="p1")]
    membership = derive_membership(products, collections)
    assert membership["women-dresses"] == ["p1"]  # matched via distinctive "dresses"
    assert membership["women-shoes"] == []  # "women" suppressed, no other overlap
    assert membership["women-bags"] == []


def test_derive_membership_allows_multi_membership_and_is_deterministic() -> None:
    collections = [
        _raw_collection_model(raw_id=1, title="Dresses", handle="dresses"),
        _raw_collection_model(raw_id=2, title="Summer", handle="summer"),
    ]
    products = [
        _raw_product_model(raw_id=1, title="Summer Dresses", handle="summer-dresses"),
        _raw_product_model(raw_id=2, title="Winter Coat", handle="winter-coat"),
    ]
    first = derive_membership(products, collections)
    second = derive_membership(products, collections)
    assert first == second
    assert first["dresses"] == ["summer-dresses"]
    assert first["summer"] == ["summer-dresses"]  # multi-membership


# --------------------------------------------------------------------------- #
# build_catalog
# --------------------------------------------------------------------------- #


def test_build_catalog_adds_catch_all_for_unmatched_products() -> None:
    products = [
        _raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses"),
        _raw_product_dict(raw_id=2, title="Mystery Widget", handle="mystery-widget"),
    ]
    collections = [_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")]
    catalog = build_catalog(
        RawProductsFeed.model_validate({"products": products}),
        RawCollectionsFeed.model_validate({"collections": collections}),
    )
    handles = [c.handle for c in catalog.collections]
    assert "all" in handles
    catch_all = next(c for c in catalog.collections if c.handle == "all")
    assert catch_all.title == "All Products"
    assert catch_all.product_handles == ["mystery-widget"]
    # Every product reachable from some collection.
    reachable = {h for c in catalog.collections for h in c.product_handles}
    assert {p.handle for p in catalog.products} <= reachable


def test_build_catalog_omits_catch_all_when_all_products_match() -> None:
    products = [_raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses")]
    collections = [_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")]
    catalog = build_catalog(
        RawProductsFeed.model_validate({"products": products}),
        RawCollectionsFeed.model_validate({"collections": collections}),
    )
    assert [c.handle for c in catalog.collections] == ["dresses"]


# --------------------------------------------------------------------------- #
# build_catalog — explicit membership override
# --------------------------------------------------------------------------- #


def test_build_catalog_uses_explicit_membership_verbatim() -> None:
    """A supplied membership map wins over the token heuristic, with no catch-all.

    Ground-truth membership mirrors the real storefront's collection set
    exactly, so an unmatched product is left in no collection rather than
    swept into a synthesized catch-all.
    """
    products = [
        _raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses"),
        _raw_product_dict(raw_id=2, title="Winter Coat", handle="winter-coat"),
    ]
    collections = [_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")]
    # The heuristic would match only "summer-dresses" on the "dresses" token;
    # the explicit map instead assigns the coat and omits the dress.
    catalog = build_catalog(
        RawProductsFeed.model_validate({"products": products}),
        RawCollectionsFeed.model_validate({"collections": collections}),
        membership={"dresses": ["winter-coat"]},
    )
    dresses = next(c for c in catalog.collections if c.handle == "dresses")
    assert dresses.product_handles == ["winter-coat"]
    # No catch-all: the unmatched dress belongs to no collection (still
    # reachable by handle / search), mirroring the real storefront.
    assert [c.handle for c in catalog.collections] == ["dresses"]


def test_build_catalog_membership_filters_unknown_product_handles() -> None:
    """Handles absent from products.json are dropped from the membership map."""
    products = [_raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses")]
    collections = [_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")]
    catalog = build_catalog(
        RawProductsFeed.model_validate({"products": products}),
        RawCollectionsFeed.model_validate({"collections": collections}),
        membership={"dresses": ["summer-dresses", "ghost-product"]},
    )
    dresses = next(c for c in catalog.collections if c.handle == "dresses")
    assert dresses.product_handles == ["summer-dresses"]
    assert [c.handle for c in catalog.collections] == ["dresses"]  # no catch-all


def test_build_catalog_drops_collection_absent_from_membership() -> None:
    """A collection with no members (absent from the map) is dropped entirely."""
    products = [_raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses")]
    collections = [
        _raw_collection_dict(raw_id=10, title="Dresses", handle="dresses"),
        _raw_collection_dict(raw_id=11, title="Shoes", handle="shoes"),
    ]
    catalog = build_catalog(
        RawProductsFeed.model_validate({"products": products}),
        RawCollectionsFeed.model_validate({"collections": collections}),
        membership={"dresses": ["summer-dresses"]},
    )
    # "shoes" has no members → dropped; only the populated "dresses" survives.
    assert [c.handle for c in catalog.collections] == ["dresses"]


def test_raw_products_collections_feed_round_trips() -> None:
    feed = RawProductsCollectionsFeed.model_validate({"dresses": ["a", "b"], "shoes": []})
    assert feed.root == {"dresses": ["a", "b"], "shoes": []}


# --------------------------------------------------------------------------- #
# IngestCatalogStep — contract
# --------------------------------------------------------------------------- #


def test_ingest_step_contract(tmp_path: Path) -> None:
    step = IngestCatalogStep(seed_dir=tmp_path / "seed")
    assert step.id == "ingest_catalog"
    assert step.phase == "data_synth"
    assert step.depends_on == []
    assert step.version == 2
    assert Path("data/products.json") in step.outputs
    assert Path("data/collections.json") in step.outputs
    assert Path(".shop_gen/stage_cache/collections.json") in step.outputs
    # Two prefetch feeds declared as file inputs.
    assert len(step.inputs) == 2


def test_ingest_step_listing_branch_has_no_inputs() -> None:
    """The ``--list-steps`` placeholder (seed_dir=None) surfaces the id only."""
    step = IngestCatalogStep(seed_dir=None)
    assert step.id == "ingest_catalog"
    assert step.inputs == []


# --------------------------------------------------------------------------- #
# IngestCatalogStep — run
# --------------------------------------------------------------------------- #


def _write_feeds(
    seed_dir: Path,
    *,
    products: list[dict[str, object]],
    collections: list[dict[str, object]],
) -> None:
    prefetch = seed_dir / "artifact" / "prefetch"
    prefetch.mkdir(parents=True)
    (prefetch / "products.json").write_text(json.dumps({"products": products}), encoding="utf-8")
    (prefetch / "collections.json").write_text(
        json.dumps({"collections": collections}),
        encoding="utf-8",
    )


def test_ingest_step_run_writes_three_valid_files(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_feeds(
        seed_dir,
        products=[
            _raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses"),
            _raw_product_dict(raw_id=2, title="Mystery Widget", handle="mystery-widget"),
        ],
        collections=[_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")],
    )
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=[seed_dir], out_dir=out_dir, catalog_source="ingest")
    IngestCatalogStep(seed_dir=seed_dir).run(
        StepContext(config=config, out_dir=out_dir, runtime=None),
    )

    products_raw = json.loads((out_dir / "data" / "products.json").read_text(encoding="utf-8"))
    collections_raw = json.loads(
        (out_dir / "data" / "collections.json").read_text(encoding="utf-8"),
    )
    drafts_raw = json.loads(
        (out_dir / ".shop_gen" / "stage_cache" / "collections.json").read_text(encoding="utf-8"),
    )

    # Data files round-trip through the closed schema.
    products = [Product.model_validate(p) for p in products_raw]
    collections = [Collection.model_validate(c) for c in collections_raw]
    assert {p.handle for p in products} == {"summer-dresses", "mystery-widget"}
    # Verbatim price + CDN src preserved.
    dress = next(p for p in products if p.handle == "summer-dresses")
    assert dress.variants[0].price == "390.00"
    assert dress.images[0].src.startswith("https://cdn.shopify.com/")
    assert dress.id == 1

    # Catch-all created for the unmatched widget; every product reachable.
    coll_handles = {c.handle for c in collections}
    assert "all" in coll_handles
    reachable = {h for c in collections for h in c.product_handles}
    assert {p.handle for p in products} <= reachable

    # The draft cache has one entry per collection (navigation reachability).
    draft_handles = {d["handle"] for d in drafts_raw}
    assert draft_handles == coll_handles


def test_ingest_step_run_is_deterministic(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_feeds(
        seed_dir,
        products=[_raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses")],
        collections=[_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")],
    )
    config = ShopGenConfig(seeds=[seed_dir], out_dir=tmp_path / "o", catalog_source="ingest")

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    IngestCatalogStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_a))
    IngestCatalogStep(seed_dir=seed_dir).run(StepContext(config=config, out_dir=out_b))
    assert (out_a / "data" / "products.json").read_text(encoding="utf-8") == (
        out_b / "data" / "products.json"
    ).read_text(encoding="utf-8")
    assert (out_a / "data" / "collections.json").read_text(encoding="utf-8") == (
        out_b / "data" / "collections.json"
    ).read_text(encoding="utf-8")


def test_ingest_step_run_missing_feed_raises(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    (seed_dir / "artifact" / "prefetch").mkdir(parents=True)
    # Only products.json present; collections.json missing.
    (seed_dir / "artifact" / "prefetch" / "products.json").write_text(
        json.dumps({"products": []}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=[seed_dir], out_dir=out_dir, catalog_source="ingest")
    with pytest.raises(StageSynthError, match=r"collections\.json"):
        IngestCatalogStep(seed_dir=seed_dir).run(
            StepContext(config=config, out_dir=out_dir),
        )


def test_ingest_step_run_without_seed_dir_raises(tmp_path: Path) -> None:
    config = ShopGenConfig(seeds=[tmp_path / "seed"], out_dir=tmp_path / "out")
    with pytest.raises(StageSynthError, match="no seed directory"):
        IngestCatalogStep(seed_dir=None).run(
            StepContext(config=config, out_dir=tmp_path / "out"),
        )


# --------------------------------------------------------------------------- #
# IngestCatalogStep — products_collections.json membership source
# --------------------------------------------------------------------------- #


def test_ingest_step_declares_membership_input_only_when_present(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_feeds(
        seed_dir,
        products=[_raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses")],
        collections=[_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")],
    )
    # Absent → two inputs only.
    assert len(IngestCatalogStep(seed_dir=seed_dir).inputs) == 2

    (seed_dir / "artifact" / "prefetch" / "products_collections.json").write_text(
        json.dumps({"dresses": ["summer-dresses"]}),
        encoding="utf-8",
    )
    # Present → the membership feed is declared as a third input.
    inputs = IngestCatalogStep(seed_dir=seed_dir).inputs
    assert len(inputs) == 3
    assert any(
        isinstance(ref, FileInput) and ref.path.name == "products_collections.json"
        for ref in inputs
    )


def test_ingest_step_run_prefers_membership_file_over_heuristic(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    _write_feeds(
        seed_dir,
        products=[
            _raw_product_dict(raw_id=1, title="Summer Dresses", handle="summer-dresses"),
            _raw_product_dict(raw_id=2, title="Winter Coat", handle="winter-coat"),
        ],
        collections=[_raw_collection_dict(raw_id=10, title="Dresses", handle="dresses")],
    )
    # Ground-truth membership assigns the coat (which the token heuristic
    # would never match to "dresses"); the dress is deliberately left out.
    (seed_dir / "artifact" / "prefetch" / "products_collections.json").write_text(
        json.dumps({"dresses": ["winter-coat"]}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    config = ShopGenConfig(seeds=[seed_dir], out_dir=out_dir, catalog_source="ingest")
    IngestCatalogStep(seed_dir=seed_dir).run(
        StepContext(config=config, out_dir=out_dir, runtime=None),
    )

    collections = [
        Collection.model_validate(c)
        for c in json.loads((out_dir / "data" / "collections.json").read_text(encoding="utf-8"))
    ]
    dresses = next(c for c in collections if c.handle == "dresses")
    assert dresses.product_handles == ["winter-coat"]
    # No catch-all in ground-truth mode: the unmatched dress is in no collection.
    assert [c.handle for c in collections] == ["dresses"]
