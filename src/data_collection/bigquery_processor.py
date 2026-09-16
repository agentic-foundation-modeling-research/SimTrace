"""
BigQuery Clickstream Extractor
==============================
Converts the real session-grain ``buyer_events_sessions`` BigQuery table
(see ``database_schema.json``) into the trajectory-tagged clickstream files the
buyer simulator consumes:

    data/sessions_raw.csv        clickstream for the sampled sessions (one row per action)
    data/sessions_raw_complete.csv  clickstream for ALL sessions (unsampled)
    data/sessions_external.csv   anonymized 10-column view (sampled)
    data/products.csv   minimal website product export (catalog source)

The unified ``src.data_collection.main`` caller enriches these collection
outputs into ``sessions.jsonl`` after this processor returns.

The source table is session-grain with nested repeated arrays
(``products[].events_timeline[]``, ``search_events[]``, ``collection_events[]``).
``src/data_collection/sql/buyer_events_to_clickstream.sql`` UNNESTs those into one flat row
per event; this script then maps each event to the simulator's ``semantic_action``
vocabulary, synthesizes dwell/terminate events, and writes the four files.

Use :mod:`src.data_collection.main` to run collection and enrichment together.

"""

import csv
import heapq
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

try:
    from .catalog import ProductCatalog
    from .enrich_sessions import aggregate_buyer_features, session_buyer_map
    from .processor import Processor
except ImportError:  # direct-file execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from catalog import ProductCatalog  # type: ignore[no-redef]
    from enrich_sessions import (  # type: ignore[no-redef]
        aggregate_buyer_features,
        session_buyer_map,
    )
    from processor import Processor  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Event -> semantic_action mapping (see plan / database_schema.json)
# ---------------------------------------------------------------------------
EVENT_TO_SEMANTIC = {
    "product_page_viewed": "detail",
    "page_viewed": "detail",
    "entity_clicked": "detail",
    "search_submitted": "explore-search",
    "collection_page_viewed": "explore-goto",
    "product_added_to_cart": "add",
    "product_removed_from_cart": "remove",
    "checkout_started": "checkout",
    "checkout_completed": "checkout",
}

# Events with no clean simulator equivalent. They are still emitted (as "Other")
# but reported so the conversion gaps are visible.
UNPROCESSABLE_EVENTS = {
    "entity_impression",
    "product_favorite_action",
    "hide_recommendation",
}

# Mirror the Clarity pipeline: a gap longer than this between consecutive events
# is approximated as a scroll/browse dwell on the preceding page.
EXPLORE_STAY_THRESHOLD_MS = 10_000

SQL_DIR = Path(__file__).resolve().parent / "sql"
SQL_PATH = SQL_DIR / "buyer_events_to_clickstream_explore.sql"
CANDIDATES_SQL_PATH = SQL_DIR / "session_candidates.sql"
SHOP_PRODUCTS_SQL_PATH = SQL_DIR / "shop_products.sql"

# Columns consumed by the simulator's product-catalog reader.
PRODUCT_EXPORT_FIELDS = [
    "product_id",
    "product_handle",
    "product_title",
    "product_body_html",
    "product_description",
    "vendor",
    "product_type",
    "tags",
    "published",
    "price",
    "status",
]


# ---------------------------------------------------------------------------
# Load events (from BigQuery or a pre-exported CSV)
# ---------------------------------------------------------------------------
def _build_where(shop_id: str | None, by_session_ids: bool = False) -> str:
    """Build the {WHERE} clause substituted into each base-table scan. When
    ``by_session_ids`` is set, also restrict to the sampled sessions passed as the
    @session_ids array query parameter (Pass B of the two-pass load)."""
    clauses = []
    if shop_id:
        clauses.append(f"s.shop_id = {shop_id}")
    if by_session_ids:
        clauses.append("s.session_id IN UNNEST(@session_ids)")
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _render_sql(table: str, shop_id: str | None, by_session_ids: bool = False) -> str:
    sql = SQL_PATH.read_text()
    return sql.replace("{TABLE}", table).replace(
        "{WHERE}", _build_where(shop_id, by_session_ids)
    )


def load_events_from_bigquery(
    project: str | None,
    table: str,
    shop_id: str | None,
    session_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Download the flat event stream. When ``session_ids`` is given, only those
    sessions are pulled (Pass B) instead of the whole shop."""
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit(
            "ERROR: google-cloud-bigquery is not installed.\n"
            "  pip install google-cloud-bigquery\n"
            "  (or use --input-csv to read a pre-exported flat CSV)"
        )
    sql = _render_sql(table, shop_id, by_session_ids=bool(session_ids))
    if session_ids:
        print(f"[1/6] Querying BigQuery for {len(session_ids)} sampled sessions ...")
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("session_ids", "STRING", session_ids)
            ]
        )
    else:
        print(f"[1/6] Querying BigQuery table {table} ...")
        job_config = None
    client = bigquery.Client(project=project)
    df = (
        client.query(sql, job_config=job_config)
        .result()
        .to_dataframe(create_bqstorage_client=False)
    )
    print(f"  -> {len(df)} event rows")
    return df


def load_session_candidates(
    project: str | None,
    table: str,
    shop_id: str | None,
    min_actions: int,
    max_actions: int,
    cap: int,
    seed: int,
) -> dict:
    """Pass A of the two-pass load: run session_candidates.sql to get one tiny
    summary row per eligible session and return the ``agg`` dict expected by
    ``select_session_ids`` ({session_id: {"outcome", "n_real", "products": set}}).
    This avoids downloading every event for the shop."""
    try:
        from google.cloud import bigquery  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: google-cloud-bigquery is not installed.\n"
            "  pip install google-cloud-bigquery\n"
            "  (or use --input-csv to read a pre-exported flat CSV)"
        )
    from google.cloud import bigquery

    sql = (
        CANDIDATES_SQL_PATH.read_text()
        .replace("{TABLE}", table)
        .replace("{WHERE}", _build_where(shop_id))
        .replace("{MIN}", str(min_actions))
        .replace("{MAX}", str(max_actions))
        .replace("{CAP}", str(cap))
        .replace("{SEED}", str(seed))
    )
    print(
        f"[1a/6] Querying candidate sessions "
        f"(<= {cap}/outcome, {min_actions}-{max_actions} real actions) ..."
    )
    client = bigquery.Client(project=project)
    df = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
    agg = {
        str(row.session_id): {
            "outcome": row.outcome,
            "n_real": int(row.n_real),
            "products": set(row.products) if row.products is not None else set(),
        }
        for row in df.itertuples(index=False)
    }
    print(f"  -> {len(agg)} candidate sessions")
    return agg


def _shop_products_df_to_dict(df: pd.DataFrame) -> dict:
    """Turn the shop_products query/cache DataFrame into
    {str(product_id): {export fields}}. Representative price is the min variant
    price, falling back to the max when there is no min."""
    products: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        r = row._asdict()
        min_price = r.get("min_price")
        max_price = r.get("max_price")
        price = min_price if pd.notna(min_price) else max_price
        # `published` is a BigQuery bool, but a round-trip through the CSV cache reads
        # it back as the string "True"/"False" (where bool("False") is truthy).
        pub = r.get("published")
        is_pub = pub.strip().lower() == "true" if isinstance(pub, str) else bool(pub)
        products[_clean(r.get("product_id"))] = {
            "product_description": _clean(r.get("product_description")),
            "product_body_html": _clean(r.get("product_body_html")),
            "product_title": _clean(r.get("product_title")),
            "product_handle": _clean(r.get("product_handle")),
            "vendor": _clean(r.get("vendor")),
            "product_type": _clean(r.get("product_type")),
            "status": _clean(r.get("status")),
            "published": "TRUE" if is_pub else "FALSE",
            "price": float(price) if pd.notna(price) else "",
        }
    return products


def load_shop_products(
    project: str | None,
    shop_id: str,
    cache_path: Path,
    refresh: bool = False,
) -> dict:
    """Return the shop's merchandising catalog as {str(product_id): {export fields}}
    for enriching products.csv with authoritative description + price.

    The underlying query is slow (~7 min/shop), so the result is cached to
    ``cache_path`` (a standalone per-shop products table) and reused on later runs
    unless ``refresh`` is set. Best-effort: on query failure the export falls back to
    the events-derived fields (returns {})."""
    if cache_path.exists() and not refresh:
        print(f"[2a/6] Reading cached shop products {cache_path} ...")
        df = pd.read_csv(cache_path, dtype={"product_id": str})
        products = _shop_products_df_to_dict(df)
        print(f"  -> {len(products)} products (cached)")
        return products

    try:
        from google.cloud import bigquery
    except ImportError:
        print("  WARNING: google-cloud-bigquery not installed; skipping product "
              "description/price enrichment")
        return {}

    sql = SHOP_PRODUCTS_SQL_PATH.read_text().replace("{SHOP_ID}", str(shop_id))
    print(f"[2a/6] Querying merchandising catalog for shop {shop_id} "
          "(slow, ~7 min; cached for reuse) ...")
    try:
        client = bigquery.Client(project=project)
        df = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:  # noqa: BLE001 - non-fatal enrichment
        print(f"  WARNING: shop products query failed ({e}); skipping "
              "description/price enrichment")
        return {}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    products = _shop_products_df_to_dict(df)
    print(f"  -> {len(products)} products -> cached to {cache_path}")
    return products


def load_events_from_csv(path: str) -> pd.DataFrame:
    print(f"[1/6] Reading flat events CSV {path} ...")
    # product_id is an integer in BigQuery; loaded as a number it becomes float64
    # whenever the column has any NaN (events with no product), so ids read back as
    # "123456.0". Read it as a string to keep the raw integer id (missing cells
    # still come in as NaN, which _clean handles).
    # collection_id is likewise an integer that becomes "1234.0" once a NaN forces
    # the column to float64; read it as a string to keep the raw id for the URL.
    df = pd.read_csv(path, dtype={"product_id": str, "collection_id": str})
    print(f"  -> {len(df)} event rows")
    return df


def _normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and drop rows we cannot place on a timeline."""
    required = {"session_id", "event_name", "event_at"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: events are missing required columns: {sorted(missing)}")
    df = df.copy()
    # BigQuery exports TIMESTAMPs with a " UTC" suffix and *variable* precision
    # (e.g. "...:09.831 UTC" vs "...:09 UTC"). format="mixed" parses each value on
    # its own; without it pandas locks onto the first row's format and coerces every
    # differently-shaped timestamp to NaT.
    df["event_at"] = pd.to_datetime(
        df["event_at"], utc=True, format="mixed", errors="coerce"
    )
    before = len(df)
    df = df.dropna(subset=["session_id", "event_at"])
    if len(df) < before:
        print(
            f"  Dropped {before - len(df)} rows with missing/unparseable "
            "session_id or event_at"
        )
    df = df.sort_values(["session_id", "event_at"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Products export + catalog
# ---------------------------------------------------------------------------
def _clean(v) -> str:
    """NaN/None -> "" (the `x or ""` idiom misses NaN, which is truthy)."""
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def build_products_export(
    df: pd.DataFrame, output_path: Path, shop_products: dict | None = None
) -> Path:
    """Write a minimal website product export from the distinct products seen.

    When ``shop_products`` (a {str(product_id): {...}} map from the merchandising
    catalog, see ``load_shop_products``) is given, each product is enriched:
    description/body_html come from the catalog (the events carry none), price is
    catalog-wins with the events-derived price as fallback, and the remaining fields
    are filled from the catalog only where the events value is empty."""
    shop_products = shop_products or {}
    prods = df[df["product_id"].notna() & (df["product_id"] != "")].copy()
    # First non-null price per product (Shop-App-only products carry no price).
    prods = prods.sort_values("price", na_position="last")
    grouped = prods.groupby("product_id", sort=False)
    rows = []
    matched = 0
    for product_id, grp in grouped:
        first = grp.iloc[0]
        price = grp["price"].dropna()
        events_price = float(price.iloc[0]) if len(price) else ""
        # product_id is the canonical identity; Handle still carries the
        # product_handle (the catalog and clickstream URL key off it).
        handle = next(
            (h for h in grp["product_handle"] if pd.notna(h) and h != ""), ""
        )
        cat = shop_products.get(_clean(product_id))
        if cat:
            matched += 1
        cat = cat or {}
        rows.append(
            {
                "product_id": product_id,
                # Fill from the catalog only when the events value is empty.
                "product_handle": handle or cat.get("product_handle", ""),
                "product_title": (
                    _clean(first.get("product_title"))
                    or cat.get("product_title", "")
                    or handle
                ),
                # Description + body_html come straight from the catalog (events
                # carry neither); fall back to any events value if the catalog misses.
                "product_body_html": (
                    cat.get("product_body_html", "")
                    or _clean(first.get("product_body_html"))
                ),
                "product_description": (
                    cat.get("product_description", "")
                    or _clean(first.get("product_description"))
                ),
                "vendor": _clean(first.get("vendor")) or cat.get("vendor", ""),
                "product_type": (
                    _clean(first.get("product_type")) or cat.get("product_type", "")
                ),
                "tags": "",
                "published": cat.get("published", "TRUE"),
                # Catalog price wins; events-derived price is the fallback.
                "price": cat.get("price") if cat.get("price") not in (None, "")
                else events_price,
                "status": cat.get("status") or "active",
            }
        )
    out = pd.DataFrame(rows, columns=PRODUCT_EXPORT_FIELDS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    suffix = (
        f"; matched {matched}/{len(out)} to merchandising catalog"
        if shop_products else ""
    )
    print(f"  -> {output_path} ({len(out)} products{suffix})")
    return output_path


# ---------------------------------------------------------------------------
# Event -> clickstream rows
# ---------------------------------------------------------------------------
def _epoch_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def _lookup(catalog, slug: str) -> tuple[str, str, str]:
    """catalog.lookup, tolerant of products with no/NaN category (the catalog
    indexes bins by category and raises KeyError on an unknown one)."""
    if not (catalog and slug):
        return "", "", ""
    try:
        return catalog.lookup(slug)
    except KeyError:
        return "", "", ""


def _row(
    processor, session_uuid, user_uuid, store_uuid, action, ts_ms, page_url,
    product_handle, catalog, raw_action, created_at, session_outcome_custom="",
    landing_url="", product_id="",
):
    p_hash = processor.hash_value(product_handle) if product_handle else ""
    p_id = _clean(product_id)
    category, bucket, price = _lookup(catalog, product_handle)
    # Shop-App-only products carry no price; keep the category but blank the
    # (meaningless) price/bucket rather than the catalog's top-bucket fallback.
    if price in ("", "nan", "None") or price.lower() == "nan":
        bucket, price = "", ""
    return {
        "session_id": session_uuid,
        "user_id": user_uuid,
        "store_id": store_uuid,
        "timestamp": ts_ms,
        "semantic_action": action,
        "raw_action": raw_action,
        "product_hash": p_hash,
        "product_id": p_id,
        "url_hash": processor.hash_value(page_url) if page_url else "",
        "url": page_url,
        "created_at": created_at,
        "persona": "",  # real data carries no persona
        "product_category": category,
        "price_bucket": bucket,
        "product_handle": product_handle,
        "price": price,
        "session_outcome_custom": session_outcome_custom,
        "landing_url": landing_url,
    }


def _transform_events_to_clickstream(
    processor: Processor,
    df: pd.DataFrame,
    catalog: ProductCatalog | None,
    explore_stay_mode: str = "r1-gap",
):
    """Convert flat events into clickstream rows. Returns (rows, unprocessable).

    explore_stay_mode controls how explore-stay rows are produced (both modes
    collapse a repeat-run of the same page into a single explore-stay and never
    emit two stays in a row):
    "r1-only" -- only the repeat-run collapse stay. The transform rules, in the
                order they apply to each event:
    
    1. Map event_name from real data -> semantic_action via EVENT_TO_SEMANTIC; an unmapped event becomes "Other".
    2. repeat -> stay: a consecutive event on the same non-empty page with the same event type is a repeat; a run of repeats collapses into one explore-stay whose timestamp extends to the latest repeat.
    3. back: returning to the page two rows back (A,B,A) becomes explore-back.
    4. Checkout is a cut point: the first checkout is emitted, then the rest of the session is discarded.
    5. Every session ends with a synthesized terminated row.

    "r1-gap"  -- all of the above, plus: synthesize an explore-stay on the
                previous page on a non-repeat transition whose time gap exceeds
                EXPLORE_STAY_THRESHOLD_MS (never adjacent to an existing stay).
    """
    print("[4/6] Transforming events to clickstream ...")
    rows: list[dict] = []
    unprocessable: list[dict] = []

    for session_id, grp in df.groupby("session_id", sort=False):
        session_uuid = str(session_id)
        client_id = next(
            (c for c in grp.get("client_id", pd.Series(dtype=str)) if pd.notna(c) and c),
            "",
        )
        user_uuid = str(client_id) if client_id else ""
        shop_id = next(
            (s for s in grp.get("shop_id", pd.Series(dtype=str)) if pd.notna(s)), ""
        )
        store_uuid = str(shop_id) if shop_id != "" else ""
        outcome_custom = _session_outcome_custom(set(grp["event_name"]))
        # landing_url is session-level (constant per session); passthrough only.
        landing_url = (
            str(grp["landing_url"].iloc[0])
            if "landing_url" in grp.columns and pd.notna(grp["landing_url"].iloc[0])
            else ""
        )

        session_rows: list[dict] = []
        prev_ts_ms: int | None = None
        prev_product_handle = ""
        prev_product_id = ""
        last_ts_ms = 0
        hit_checkout = False
        # Previous two raw events' page URLs + previous event name, for the
        # URL-based explore-stay (R1) / explore-back (R2) relabeling below.
        prev1_url = ""
        prev1_event = ""
        prev2_url = ""

        for _, ev in grp.iterrows():
            event_name = ev["event_name"]
            action = EVENT_TO_SEMANTIC.get(event_name, "Other")
            product_handle = ev.get("product_handle") or ""
            product_handle = "" if pd.isna(product_handle) else str(product_handle)
            product_id = _clean(ev.get("product_id"))
            query = _clean(ev.get("query"))
            collection_handle = _clean(ev.get("collection_handle"))
            collection_id = _clean(ev.get("collection_id"))
            ts_ms = _epoch_ms(ev["event_at"])
            created_at = ev["event_at"].isoformat()
            # Synthesize the page URL per event kind: product pages key off the
            # handle, search events off the ?q= term, collection pages off the
            # collection handle (preferred) or id (INT64 -> may read as "123.0").
            if product_handle:
                page_url = f"/products/{product_handle}"
            elif action == "explore-search" and query:
                page_url = f"/search?q={quote_plus(query)}"
            elif action == "explore-goto" and (collection_handle or collection_id):
                # Strip a trailing ".0" left when the INT64 id rides through a
                # float column (the direct-BigQuery path doesn't reload from CSV).
                coll = collection_handle or collection_id.removesuffix(".0")
                page_url = f"/collections/{coll}"
            else:
                page_url = ""

            # A consecutive event on the same (non-empty) page with the same event
            # type is a repeat-view -- a run of these collapses into one stay.
            is_repeat = (
                bool(page_url)
                and page_url == prev1_url
                and event_name == prev1_event
            )

            if action == "Other":
                unprocessable.append(
                    {
                        "session_id": str(session_id),
                        "event_name": event_name,
                        "event_source": ev.get("event_source", ""),
                        "product_handle": product_handle,
                        "event_at": created_at,
                    }
                )
            elif is_repeat:
                # R1: repeat of the previous page -> dwell.
                action = "explore-stay"
            elif page_url and page_url == prev2_url and page_url != prev1_url:
                # R2: A,B,A return to the page two rows back -> back-navigation.
                action = "explore-back"

            last_emitted = (
                session_rows[-1]["semantic_action"] if session_rows else None
            )

            # r1-gap only: synthesize an explore-stay on the previous page when the
            # user lingered (>threshold) across a non-repeat transition -- but never
            # adjacent to an existing stay (keeps stays from doubling up).
            if (
                explore_stay_mode == "r1-gap"
                and not is_repeat
                and prev_ts_ms is not None
                and ts_ms - prev_ts_ms > EXPLORE_STAY_THRESHOLD_MS
                and action != "explore-stay"
                and last_emitted != "explore-stay"
            ):
                stay_url = (
                    f"/products/{prev_product_handle}" if prev_product_handle else ""
                )
                session_rows.append(
                    _row(
                        processor, session_uuid, user_uuid, store_uuid, "explore-stay",
                        prev_ts_ms + 1, stay_url, prev_product_handle, catalog,
                        processor.build_raw_action(
                            "explore-stay", stay_url, prev_product_handle
                        ),
                        ev["event_at"].isoformat(), outcome_custom, landing_url,
                        prev_product_id,
                    )
                )
                last_emitted = "explore-stay"

            if is_repeat and last_emitted == "explore-stay":
                # Collapse the rest of a repeat-run into the single existing stay,
                # extending its timestamp to this (latest) repeat.
                session_rows[-1]["timestamp"] = ts_ms
                session_rows[-1]["created_at"] = created_at
            else:
                raw_action = (
                    json.dumps(
                        {"event_name": event_name, "source": ev.get("event_source", "")}
                    )
                    if action == "Other"
                    else processor.build_raw_action(action, page_url, product_handle)
                )
                session_rows.append(
                    _row(
                        processor, session_uuid, user_uuid, store_uuid, action, ts_ms, page_url,
                        product_handle, catalog, raw_action, created_at, outcome_custom,
                        landing_url, product_id,
                    )
                )

            prev_ts_ms, prev_product_handle, prev_product_id, last_ts_ms = (
                ts_ms, product_handle, product_id, ts_ms
            )
            prev2_url, prev1_url, prev1_event = prev1_url, page_url, event_name

            # Checkout is a cut point: emit it, then discard everything after.
            if action == "checkout":
                hit_checkout = True
                break

        # Always terminate the session.
        term_ts = last_ts_ms + 1
        term_created = pd.Timestamp(term_ts, unit="ms", tz="UTC").isoformat()
        session_rows.append(
            _row(
                processor, session_uuid, user_uuid, store_uuid, "terminate", term_ts, "", "",
                catalog, processor.build_raw_action("terminate", "", ""), term_created,
                outcome_custom, landing_url,
            )
        )
        rows.extend(session_rows)

    print(
        f"  Generated {len(rows)} clickstream rows from "
        f"{df['session_id'].nunique()} sessions"
    )
    return rows, unprocessable


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
_CHECKOUT_EVENTS = {"checkout_started", "checkout_completed"}

# Action-derived outcome classes (see _session_outcome_custom). Note this differs
# from the declared `session_outcome` column: a *started-but-not-completed*
# checkout is labeled "checkout" here, whereas the declared column reserves
# "purchase" for completed checkouts only.
OUTCOMES = ["checkout", "cart_abandoned", "browse_only"]


def _session_outcome_custom(event_names: set[str]) -> str:
    """`session_outcome_custom`: any checkout event -> "checkout"; else an
    add-to-cart -> "cart_abandoned"; else "browse_only"."""
    if event_names & _CHECKOUT_EVENTS:
        return "checkout"
    if "product_added_to_cart" in event_names:
        return "cart_abandoned"
    return "browse_only"


def _truncate_at_checkout(grp: pd.DataFrame) -> pd.DataFrame:
    """Mirror transform_to_clickstream's cut point: keep events up to and
    including the first checkout, dropping everything after (those events are
    never emitted, so they must not count toward eligibility or coverage)."""
    is_checkout = grp["event_name"].isin(_CHECKOUT_EVENTS).to_numpy()
    if is_checkout.any():
        first = int(is_checkout.argmax())  # position of the first checkout
        return grp.iloc[: first + 1]
    return grp


def _aggregate_sessions(events: pd.DataFrame) -> dict:
    """Per-session: the action-derived `session_outcome_custom`, the count of
    real interaction events, and the set of non-null product_ids touched -- the
    latter two measured on the events the transform will actually emit (i.e.
    truncated at the first checkout)."""
    has_pid = "product_id" in events.columns
    agg: dict[str, dict] = {}
    for session_id, grp in events.groupby("session_id", sort=False):
        names = set(grp["event_name"])
        emitted = _truncate_at_checkout(grp)
        products = (
            set(emitted["product_id"].dropna().tolist()) if has_pid else set()
        )
        agg[session_id] = {
            "outcome": _session_outcome_custom(names),
            "n_real": int(emitted["event_name"].isin(EVENT_TO_SEMANTIC).sum()),
            "products": products,
        }
    return agg


def select_session_ids(
    agg: dict,
    n: int,
    min_actions: int,
    max_actions: int,
    seed: int,
) -> list[str]:
    """Pick ``n`` session_ids split evenly across session_outcome, drawing the
    sessions that together cover the most distinct product_ids (greedy
    set-cover), restricted to sessions with ``min_actions``..``max_actions`` real
    interaction events.

    ``agg`` maps session_id -> {"outcome", "n_real", "products": set}. It can be
    built from an events DataFrame (``_aggregate_sessions``) or straight from the
    Pass A candidate query (``load_session_candidates``)."""
    print(f"\n[sample] Selecting {n} sessions "
          f"({min_actions}-{max_actions} real actions, even outcome split, "
          f"max product diversity) ...")
    rng = random.Random(seed)

    # Eligibility: between min and max real interaction events.
    eligible = {
        sid: info
        for sid, info in agg.items()
        if min_actions <= info["n_real"] <= max_actions
    }

    # Group eligible session_ids by outcome.
    by_outcome: dict[str, list[str]] = {o: [] for o in OUTCOMES}
    for sid, info in eligible.items():
        by_outcome.setdefault(info["outcome"], []).append(sid)

    # Quotas: even split, distribute any remainder, then cap by availability and
    # redistribute the shortfall so the total still reaches n.
    base, rem = divmod(n, len(OUTCOMES))
    quota = {o: base + (1 if i < rem else 0) for i, o in enumerate(OUTCOMES)}
    shortfall = 0
    for o in OUTCOMES:
        avail = len(by_outcome[o])
        if avail < quota[o]:
            shortfall += quota[o] - avail
            print(f"[sample]   WARNING: only {avail} eligible '{o}' sessions, "
                  f"wanted {quota[o]}")
            quota[o] = avail
    # Redistribute shortfall to strata that still have spare eligible sessions.
    for o in OUTCOMES:
        if shortfall <= 0:
            break
        spare = len(by_outcome[o]) - quota[o]
        take = min(spare, shortfall)
        quota[o] += take
        shortfall -= take

    # Global greedy max-coverage with per-stratum quotas (lazy evaluation).
    covered: set = set()
    selected: list[str] = []
    remaining = dict(quota)
    # Deterministic tie-break order independent of dict iteration / data order.
    order = {sid: i for i, sid in enumerate(sorted(eligible))}
    # Max-heap via negative gain: (-gain, order_key, sid).
    heap = [
        (-len(eligible[sid]["products"]), order[sid], sid) for sid in eligible
    ]
    heapq.heapify(heap)
    while heap and sum(remaining.values()) > 0:
        neg_gain, okey, sid = heapq.heappop(heap)
        outcome = eligible[sid]["outcome"]
        if remaining.get(outcome, 0) <= 0:
            continue  # stratum already full -> drop
        gain = len(eligible[sid]["products"] - covered)
        if gain == 0 and -neg_gain != 0:
            # was non-zero when pushed but now fully covered -> reinsert as 0 so
            # it competes fairly with other zero-gain sessions during fill.
            heapq.heappush(heap, (0, okey, sid))
            continue
        if -neg_gain != gain:
            heapq.heappush(heap, (-gain, okey, sid))  # stale -> refresh
            continue
        selected.append(sid)
        covered |= eligible[sid]["products"]
        remaining[outcome] -= 1

    # Random-fill any leftover quota (reached only if a stratum ran dry above).
    if sum(remaining.values()) > 0:
        chosen = set(selected)
        for o in OUTCOMES:
            if remaining[o] <= 0:
                continue
            pool = [s for s in by_outcome[o] if s not in chosen]
            rng.shuffle(pool)
            take = pool[: remaining[o]]
            selected.extend(take)
            remaining[o] -= len(take)

    _report_sample(eligible, by_outcome, selected, covered)
    return selected


def sample_sessions(
    events: pd.DataFrame,
    n: int,
    min_actions: int,
    max_actions: int,
    seed: int,
) -> pd.DataFrame:
    """Sample ``n`` sessions from an in-memory events DataFrame (the --input-csv
    path). Aggregates per-session stats, picks the session_ids, and returns
    ``events`` filtered to the chosen sessions."""
    agg = _aggregate_sessions(events)
    selected = select_session_ids(agg, n, min_actions, max_actions, seed)
    return events[events["session_id"].isin(selected)].reset_index(drop=True)


def _report_sample(eligible, by_outcome, selected, covered) -> None:
    sel_set = set(selected)
    total_products = set()
    for info in eligible.values():
        total_products |= info["products"]
    no_product = sum(1 for s in selected if not eligible[s]["products"])
    print(f"[sample]   eligible sessions: {len(eligible)}")
    for o in OUTCOMES:
        elig = len(by_outcome[o])
        sel = sum(1 for s in by_outcome[o] if s in sel_set)
        print(f"[sample]     {o:<16} eligible={elig:<7} selected={sel}")
    print(f"[sample]   total selected: {len(selected)}")
    print(f"[sample]   distinct product_id covered: {len(covered)} "
          f"of {len(total_products)} available in eligible pool")
    print(f"[sample]   selected sessions with no product: {no_product}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def report_statistics(
    events: pd.DataFrame, rows: list[dict], title: str = "Data statistics"
) -> list[str]:
    """Build the data-statistics report as markdown lines, print them to the
    console, and return them so the caller can also write them to a file."""
    out: list[str] = [f"## {title}"]

    # 1. session count
    n_sessions = events["session_id"].nunique()
    out += ["", "### 1. Sessions", "", f"- Total sessions: **{n_sessions}**"]

    # 2. events per session
    per_session = events.groupby("session_id").size()
    out += [
        "",
        "### 2. Events per session",
        "",
        f"- mean = {per_session.mean():.2f}",
        f"- std = {per_session.std():.2f}",
        f"- min = {int(per_session.min())}",
        f"- median = {per_session.median():.1f}",
        f"- max = {int(per_session.max())}",
    ]

    # percentiles
    pctls = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    qs = per_session.quantile([p / 100 for p in pctls])
    out += ["", "#### Percentiles", "", "| percentile | events |", "| --- | --- |"]
    for p in pctls:
        out.append(f"| p{p} | {qs[p / 100]:.1f} |")

    # histogram by bucket
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float("inf")]
    labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    binned = pd.cut(per_session, bins=bins, labels=labels, right=True)
    counts = binned.value_counts().reindex(labels, fill_value=0)
    out += [
        "",
        "#### Distribution",
        "",
        "| bucket | count | % | |",
        "| --- | --- | --- | --- |",
    ]
    max_cnt = int(counts.max()) or 1
    for label, cnt in counts.items():
        bar = "#" * int(cnt / max_cnt * 40)
        pct = cnt / n_sessions * 100
        out.append(f"| {label} | {int(cnt)} | {pct:.1f}% | {bar} |")

    # 3. event-type and semantic-action breakdowns
    out += ["", "### 3a. Raw event_name counts", "", "| event_name | count |", "| --- | --- |"]
    for name, cnt in events["event_name"].value_counts().items():
        out.append(f"| {name} | {cnt} |")
    out += [
        "",
        "### 3b. Mapped semantic_action counts (emitted rows)",
        "",
        "| semantic_action | count |",
        "| --- | --- |",
    ]
    sem = pd.Series([r["semantic_action"] for r in rows])
    for name, cnt in sem.value_counts().items():
        out.append(f"| {name} | {cnt} |")

    # 4. session outcome
    out += ["", "### 4. Session outcome", ""]
    if "session_outcome" in events.columns:
        outcomes = events.drop_duplicates("session_id")["session_outcome"]
        counts = outcomes.value_counts(dropna=False)
        out += ["| outcome | count | % |", "| --- | --- | --- |"]
        for name, cnt in counts.items():
            out.append(f"| {name} | {cnt} | {cnt / n_sessions * 100:.1f}% |")
    else:
        out.append("_(session_outcome column not present)_")

    # 4b. action-derived session_outcome_custom (the column we emit/balance on)
    out += [
        "",
        "### 4b. session_outcome_custom (action-derived)",
        "",
        "| outcome | count | % |",
        "| --- | --- | --- |",
    ]
    so = pd.Series([r["session_outcome_custom"] for r in rows], dtype=str)
    sid = pd.Series([r["session_id"] for r in rows])
    per_session_outcome = so.groupby(sid).first()
    for name, cnt in per_session_outcome.value_counts().items():
        out.append(f"| {name} | {cnt} | {cnt / n_sessions * 100:.1f}% |")

    print("\n" + "\n".join(out) + "\n")
    return out


def report_unprocessable(unprocessable: list[dict], output_dir: Path) -> list[str]:
    """Build the unprocessable-events report as markdown lines, print them, write
    the full list to CSV, and return the lines for the combined report file."""
    out: list[str] = ["## Unprocessable events (mapped to 'Other')"]
    if not unprocessable:
        out += ["", "None -- every event mapped to a known semantic_action."]
        print("\n" + "\n".join(out) + "\n")
        return out
    udf = pd.DataFrame(unprocessable)
    out += [
        "",
        f"Total: **{len(udf)}** events across {udf['session_id'].nunique()} sessions",
        "",
        "### By event_name",
        "",
        "| event_name | count | note |",
        "| --- | --- | --- |",
    ]
    for name, cnt in udf["event_name"].value_counts().items():
        note = "no simulator equivalent" if name in UNPROCESSABLE_EVENTS else ""
        out.append(f"| {name} | {cnt} | {note} |")
    out += [
        "",
        "### Sample rows",
        "",
        "```",
        udf.head(10).to_string(index=False),
        "```",
    ]
    csv_out = output_dir / "unprocessable_events.csv"
    udf.to_csv(csv_out, index=False)
    out += ["", f"_Full list written to {csv_out}_"]

    print("\n" + "\n".join(out) + "\n")
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
BIGQUERY_RAW_FIELDS = [
    "session_id",
    "user_id",
    "store_id",
    "timestamp",
    "semantic_action",
    "raw_action",
    "product_hash",
    "product_id",
    "url_hash",
    "url",
    "created_at",
    "persona",
    "product_category",
    "price_bucket",
    "product_handle",
    "price",
    "session_outcome_custom",
    "landing_url",
]


def write_raw_csv(
    rows: list[dict], path: Path, fields: list[str] = BIGQUERY_RAW_FIELDS
):
    """Write clickstream ``rows`` to ``path`` with the given column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class BigQueryProcessor(Processor):
    """Standardize BigQuery events using the shared :class:`Processor` contract.

    BigQuery-only configuration belongs in ``meta_arg``. Supported keys are
    ``project``, ``table``, ``shop_id``, ``input_csv``, ``sample_sessions``,
    ``candidate_cap``, ``min_actions``, ``max_actions``, ``seed``, and
    ``explore_stay_mode``.
    """

    source_name = "bigquery"
    raw_fields = BIGQUERY_RAW_FIELDS

    def fetch(self) -> pd.DataFrame:
        """Load and optionally sample events according to ``meta_arg``."""
        project = self.get_meta_arg("project")
        table = self.get_meta_arg("table")
        shop_id = self.get_meta_arg("shop_id")
        input_csv = self.get_meta_arg("input_csv")
        sample_count = self.get_meta_arg("sample_sessions")
        min_actions = self.get_meta_arg("min_actions", 3)
        max_actions = self.get_meta_arg("max_actions", 20)
        seed = self.get_meta_arg("seed", 42)

        self.sampled_via_sql = False
        if input_csv:
            events = load_events_from_csv(input_csv)
        else:
            if not table:
                raise ValueError("table is required when input_csv is not provided")
            if sample_count:
                agg = load_session_candidates(
                    project,
                    table,
                    shop_id,
                    min_actions,
                    max_actions,
                    self.get_meta_arg("candidate_cap", 50_000),
                    seed,
                )
                if not agg:
                    raise ValueError(
                        "no candidate sessions matched the eligibility filter "
                        f"({min_actions}-{max_actions} real actions)"
                    )
                selected = select_session_ids(
                    agg, sample_count, min_actions, max_actions, seed
                )
                events = load_events_from_bigquery(
                    project, table, shop_id, session_ids=selected
                )
                self.sampled_via_sql = True
            else:
                events = load_events_from_bigquery(project, table, shop_id)

        events = _normalize_events(events)
        if events.empty:
            raise ValueError("no events to process")
        return events

    def transform_to_clickstream(self, raw: pd.DataFrame) -> list[dict]:
        """Map normalized BigQuery rows to the common clickstream schema."""
        catalog = self.require_catalog()
        rows, self.unprocessable = _transform_events_to_clickstream(
            self,
            raw,
            catalog,
            explore_stay_mode=self.get_meta_arg("explore_stay_mode", "r1-gap"),
        )
        return rows

    def run(self, output_dir: Path) -> list[dict]:
        """Run BigQuery's catalog, sampling, reporting, and output workflow.

        Enrichment deliberately remains outside the processor. The unified
        :mod:`src.data_collection.main` caller invokes it after this method,
        exactly as it does for every other provider.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        events = self.fetch()
        result_name = (
            f"bq-results-{self.get_meta_arg('shop_id')}.csv"
            if self.get_meta_arg("shop_id")
            else "bq-results.csv"
        )
        events.to_csv(output_dir / result_name, index=False)

        # Keep the complete event set separate from the sampled simulator set.
        if self.sampled_via_sql:
            print("[1c] Pulling ALL sessions for the complete clickstream ...")
            full_events = _normalize_events(
                load_events_from_bigquery(
                    self.get_meta_arg("project"),
                    self.get_meta_arg("table"),
                    self.get_meta_arg("shop_id"),
                )
            )
        else:
            full_events = events
            if self.get_meta_arg("sample_sessions"):
                events = sample_sessions(
                    events,
                    self.get_meta_arg("sample_sessions"),
                    self.get_meta_arg("min_actions", 3),
                    self.get_meta_arg("max_actions", 20),
                    self.get_meta_arg("seed", 42),
                )

        shop_products = None
        if self.get_meta_arg("shop_id") and not self.get_meta_arg("input_csv"):
            shop_products = load_shop_products(
                self.get_meta_arg("project"),
                self.get_meta_arg("shop_id"),
                output_dir / f"shop_products_{self.get_meta_arg('shop_id')}.csv",
                refresh=self.get_meta_arg("refresh_shop_products", False),
            )
        print("[2/6] Building products export ...")
        products_path = build_products_export(
            events, output_dir / "products.csv", shop_products
        )

        print("[3/6] Building product catalog ...")
        self.catalog = ProductCatalog(str(products_path))

        rows = self.transform_to_clickstream(events)
        sampled = len(full_events) != len(events)
        if sampled:
            full_rows, _ = _transform_events_to_clickstream(
                self,
                full_events,
                self.catalog,
                explore_stay_mode=self.get_meta_arg("explore_stay_mode", "r1-gap"),
            )
        else:
            full_rows = rows

        if sampled:
            stats_lines = (
                report_statistics(
                    full_events, full_rows, title="Full dataset (pre-sample)"
                )
                + [""]
                + report_statistics(events, rows, title="Sampled subset")
            )
        else:
            stats_lines = report_statistics(events, rows)
        unproc_lines = report_unprocessable(self.unprocessable, output_dir)
        header = [
            "# Clickstream extraction report",
            "",
            f"- shop_id: {self.get_meta_arg('shop_id') or '(all)'}",
            f"- output dir: {output_dir}",
            "",
        ]
        report_path = output_dir / "statistics_report.md"
        report_path.write_text(
            "\n".join(header + stats_lines + [""] + unproc_lines) + "\n",
            encoding="utf-8",
        )
        print(f"  -> statistics report written to {report_path}")

        print("[5/6] Writing clickstream CSVs ...")
        self.write_clickstream_csv(rows, output_dir)
        complete_path = output_dir / "sessions_raw_complete.csv"
        write_raw_csv(full_rows, complete_path)
        print(f"  -> {complete_path} ({len(full_rows)} rows, all sessions)")

        buyer_features = aggregate_buyer_features(events)
        self._persona_id_by_session = session_buyer_map(events)
        features_path = output_dir / "buyers_feature.json"
        features_path.write_text(
            json.dumps(buyer_features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  -> {features_path} ({len(buyer_features)} buyers)")
        return rows

    def persona_id_by_session(self, rows: list[dict]) -> dict[str, str]:
        """Use BigQuery's richer client/shop buyer key for enrichment."""
        if hasattr(self, "_persona_id_by_session"):
            return self._persona_id_by_session
        return super().persona_id_by_session(rows)
