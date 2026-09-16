"""
Enrich standardized clickstream data for the buyer simulation pipeline.

Takes the real clickstream (data/sessions_external.csv) then outputs:
  1. data/anonymized_clickstream.csv   — anonymized data
  2. data/sessions.jsonl    — per-session trajectories + intents for the batch simulator

Usage:
    python -m src.data_collection.enrich_sessions \
        --input data/sessions_external.csv \
        --csv-out data/anonymized_clickstream.csv \
        --jsonl-out data/sessions.jsonl \
        --features data/buyers_feature.json
"""

import argparse

import pandas as pd

try:
    from .generate_personas import (
        DEFAULT_BASE_URL,
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        generate_personas,
    )
except ImportError:  # direct-file execution fallback
    from generate_personas import (  # type: ignore[no-redef]
        DEFAULT_BASE_URL,
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        generate_personas,
    )


_CHECKOUT_EVENTS = {"checkout_started", "checkout_completed"}
_PRODUCT_VIEW_EVENTS = {"product_page_viewed", "page_viewed", "entity_clicked"}
BROWSED_PRODUCTS_CAP = 50


def _clean_value(value) -> str:
    """Normalize optional scalar values for JSON-safe feature records."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def session_buyer_map(events: pd.DataFrame) -> dict[str, str]:
    """Map sessions to stable buyer keys derived from client and website ids."""
    has_client = "client_id" in events.columns
    has_website = "shop_id" in events.columns
    result: dict[str, str] = {}
    for session_id, group in events.groupby("session_id", sort=False):
        client_id = (
            next(
                (
                    str(value)
                    for value in group["client_id"]
                    if pd.notna(value) and str(value) != ""
                ),
                "",
            )
            if has_client
            else ""
        )
        website_id = (
            next(
                (
                    str(value)
                    for value in group["shop_id"]
                    if pd.notna(value) and str(value) != ""
                ),
                "",
            )
            if has_website
            else ""
        )
        result[str(session_id)] = (
            f"{client_id}-{website_id}" if client_id else str(session_id)
        )
    return result


def _first_price(prices: pd.Series) -> float:
    values = prices.dropna()
    return round(float(values.iloc[0]), 2) if len(values) else 0


def _collect_products(rows: pd.DataFrame, with_quantity: bool) -> list[dict]:
    """Collect distinct products for persona enrichment."""
    if rows.empty or "product_id" not in rows.columns:
        return []
    products: list[dict] = []
    for product_id, group in rows.groupby("product_id", sort=False):
        if not _clean_value(product_id):
            continue
        first = group.iloc[0]
        title = _clean_value(first.get("product_title"))
        if not title:
            continue
        product = {
            "title": title,
            "price_usd": _first_price(group["_price"]),
            "description": _clean_value(first.get("product_description")),
        }
        if with_quantity:
            quantities = group["_qty"].dropna()
            product["quantity"] = int(quantities.iloc[0]) if len(quantities) else 1
        products.append(product)
    return products


def aggregate_buyer_features(events: pd.DataFrame) -> dict[str, dict]:
    """Aggregate normalized events into per-buyer persona input features."""
    data = events.copy()
    data["buyer_id"] = data["session_id"].astype(str).map(session_buyer_map(data))
    data["_price"] = (
        pd.to_numeric(data["price"], errors="coerce")
        if "price" in data.columns
        else pd.Series(pd.NA, index=data.index, dtype="float64")
    )
    data["_qty"] = (
        pd.to_numeric(data["quantity"], errors="coerce").fillna(1)
        if "quantity" in data.columns
        else pd.Series(1.0, index=data.index)
    )

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0

    features: dict[str, dict] = {}
    for buyer_id, buyer_group in data.groupby("buyer_id", sort=False):
        durations, searches, views, distinct_views, collections = [], [], [], [], []
        max_cart_values, order_values = [], []
        has_view = has_cart = has_checkout = 0
        sessions = list(buyer_group.groupby("session_id", sort=False))

        for _, session_group in sessions:
            names = session_group["event_name"]
            timestamps = session_group["event_at"]
            durations.append((timestamps.max() - timestamps.min()).total_seconds())
            searches.append(int((names == "search_submitted").sum()))
            view_mask = names.isin(_PRODUCT_VIEW_EVENTS)
            views.append(int(view_mask.sum()))
            viewed_ids = session_group.loc[view_mask, "product_id"].map(_clean_value)
            distinct_views.append(int(viewed_ids[viewed_ids != ""].nunique()))
            collections.append(int((names == "collection_page_viewed").sum()))
            has_view += int(view_mask.any())

            cart_mask = names == "product_added_to_cart"
            if cart_mask.any():
                has_cart += 1
                max_cart_values.append(
                    float(
                        (
                            session_group.loc[cart_mask, "_price"].fillna(0)
                            * session_group.loc[cart_mask, "_qty"]
                        ).sum()
                    )
                )
            has_checkout += int((names == "checkout_completed").any())
            checkout_mask = names.isin(_CHECKOUT_EVENTS)
            if checkout_mask.any():
                order_values.append(
                    float(
                        (
                            session_group.loc[checkout_mask, "_price"].fillna(0)
                            * session_group.loc[checkout_mask, "_qty"]
                        ).sum()
                    )
                )

        session_count = len(sessions)
        features[str(buyer_id)] = {
            "unique_id": str(buyer_id),
            "total_sessions": session_count,
            "avg_session_duration_seconds": average(durations),
            "avg_number_of_searches": average(searches),
            "avg_number_of_product_views": average(views),
            "avg_distinct_products_viewed": average(distinct_views),
            "avg_number_of_collection_views": average(collections),
            "product_view_rate": (
                round(has_view / session_count, 2) if session_count else 0
            ),
            "add_to_cart_rate": (
                round(has_cart / session_count, 2) if session_count else 0
            ),
            "checkout_complete_rate": (
                round(has_checkout / session_count, 2) if session_count else 0
            ),
            "avg_max_cart_value_usd": average(max_cart_values),
            "avg_order_value_usd": average(order_values),
            "browsed_products": _collect_products(
                buyer_group[buyer_group["event_name"].isin(_PRODUCT_VIEW_EVENTS)],
                with_quantity=False,
            )[:BROWSED_PRODUCTS_CAP],
            "checkout_products": _collect_products(
                buyer_group[buyer_group["event_name"].isin(_CHECKOUT_EVENTS)],
                with_quantity=True,
            ),
        }
    return features


def _clickstream_product_key(row: pd.Series) -> str:
    for field in ("product_hash", "product_slug", "product_handle", "product_id"):
        value = _clean_value(row.get(field))
        if value:
            return value
    return ""


def _clickstream_products(rows: pd.DataFrame) -> list[dict]:
    """Build persona-prompt product summaries from standardized clickstream rows."""
    products: list[dict] = []
    seen: set[str] = set()
    for _, row in rows.iterrows():
        key = _clickstream_product_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        title = (
            _clean_value(row.get("product_slug"))
            or _clean_value(row.get("product_handle"))
            or _clean_value(row.get("product_category"))
        )
        if not title:
            continue
        price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
        products.append(
            {
                "title": title,
                "price_usd": round(float(price), 2) if pd.notna(price) else 0,
                "description": "",
            }
        )
    return products


def aggregate_clickstream_buyer_features(
    rows: list[dict] | pd.DataFrame,
) -> dict[str, dict]:
    """Aggregate provider-standardized rows into persona input features.

    This is the provider-neutral fallback used by Clarity and PostHog. The
    BigQuery processor retains :func:`aggregate_buyer_features` because its raw
    event schema contains richer product descriptions and quantities.
    """
    data = pd.DataFrame(rows).copy()
    if data.empty:
        return {}

    def buyer_id_for_group(group: pd.DataFrame) -> str:
        personas = [
            _clean_value(value)
            for value in group.get("persona", pd.Series(dtype=object))
            if _clean_value(value)
        ]
        if personas:
            return personas[0]
        user_id = next(
            (
                _clean_value(value)
                for value in group.get("user_id", pd.Series(dtype=object))
                if _clean_value(value)
            ),
            "",
        )
        store_id = next(
            (
                _clean_value(value)
                for value in group.get("store_id", pd.Series(dtype=object))
                if _clean_value(value)
            ),
            "",
        )
        session_id = _clean_value(group["session_id"].iloc[0])
        return f"{user_id}-{store_id}" if user_id else session_id

    buyer_by_session = {
        str(session_id): buyer_id_for_group(group)
        for session_id, group in data.groupby("session_id", sort=False)
    }
    data["buyer_id"] = data["session_id"].astype(str).map(buyer_by_session)
    data["_timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
    data["_price"] = pd.to_numeric(
        data.get("price", pd.Series(index=data.index, dtype=float)),
        errors="coerce",
    )

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0

    features: dict[str, dict] = {}
    for buyer_id, buyer_group in data.groupby("buyer_id", sort=False):
        durations, searches, views, distinct_views, collections = [], [], [], [], []
        cart_values, order_values = [], []
        has_view = has_cart = has_checkout = 0
        checkout_product_rows: list[pd.DataFrame] = []
        sessions = list(buyer_group.groupby("session_id", sort=False))

        for _, session_group in sessions:
            actions = session_group["semantic_action"].fillna("")
            timestamps = session_group["_timestamp"].dropna()
            duration_ms = timestamps.max() - timestamps.min() if len(timestamps) else 0
            durations.append(float(duration_ms) / 1000)
            searches.append(int((actions == "explore-search").sum()))
            detail_mask = actions == "detail"
            views.append(int(detail_mask.sum()))
            product_keys = session_group.loc[detail_mask].apply(
                _clickstream_product_key, axis=1
            )
            distinct_views.append(int(product_keys[product_keys != ""].nunique()))
            collections.append(int((actions == "explore-goto").sum()))
            has_view += int(detail_mask.any())

            add_mask = actions == "add"
            if add_mask.any():
                has_cart += 1
                cart_values.append(
                    float(session_group.loc[add_mask, "_price"].fillna(0).sum())
                )

            checked_out = (actions == "checkout").any()
            has_checkout += int(checked_out)
            if checked_out:
                purchased = session_group.loc[add_mask]
                if purchased.empty:
                    purchased = session_group.loc[detail_mask].tail(1)
                checkout_product_rows.append(purchased)
                order_values.append(float(purchased["_price"].fillna(0).sum()))

        session_count = len(sessions)
        browsed_rows = buyer_group[
            buyer_group["semantic_action"].isin(["detail", "add"])
        ]
        checkout_rows = (
            pd.concat(checkout_product_rows, ignore_index=True)
            if checkout_product_rows
            else pd.DataFrame(columns=buyer_group.columns)
        )
        checkout_products = _clickstream_products(checkout_rows)
        for product in checkout_products:
            product["quantity"] = 1

        features[str(buyer_id)] = {
            "unique_id": str(buyer_id),
            "total_sessions": session_count,
            "avg_session_duration_seconds": average(durations),
            "avg_number_of_searches": average(searches),
            "avg_number_of_product_views": average(views),
            "avg_distinct_products_viewed": average(distinct_views),
            "avg_number_of_collection_views": average(collections),
            "product_view_rate": (
                round(has_view / session_count, 2) if session_count else 0
            ),
            "add_to_cart_rate": (
                round(has_cart / session_count, 2) if session_count else 0
            ),
            "checkout_complete_rate": (
                round(has_checkout / session_count, 2) if session_count else 0
            ),
            "avg_max_cart_value_usd": average(cart_values),
            "avg_order_value_usd": average(order_values),
            "browsed_products": _clickstream_products(browsed_rows)[
                :BROWSED_PRODUCTS_CAP
            ],
            "checkout_products": checkout_products,
        }
    return features


def to_anonymized_csv(df: pd.DataFrame, output_path: str):
    """Write the compact anonymized clickstream CSV."""
    columns = [
        "session_id",
        "user_id",
        "store_id",
        "timestamp",
        "semantic_action",
        "url_hash",
        "product_category",
        "price_bucket",
    ]
    out = df[columns].copy()
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {len(out)} rows ({out['session_id'].nunique()} sessions) to {output_path}"
    )
    return out


def to_sessions_jsonl(df: pd.DataFrame, output_path: str, persona_id_by_session=None):
    """Convert row-level clickstream to per-session JSONL for the simulator.

    When ``persona_id_by_session`` (a ``session_id -> buyer_id`` mapping) is
    provided, each record gets a ``persona_id`` (the buyer key) and an empty
    ``persona`` placeholder for ``generate_personas.py`` to fill later.
    """
    df = df.sort_values(["session_id", "timestamp"])
    records = []
    for sid, grp in df.groupby("session_id", sort=False):
        trajectory = grp["semantic_action"].tolist()
        cats = [c for c in grp["product_category"].dropna().unique() if c]
        buckets = grp["price_bucket"].dropna()
        buckets = [int(b) for b in buckets.unique() if str(b).strip()]
        intent = (
            "I'm looking for " + ", ".join(cats) if cats else "I'm just looking around"
        )
        record = {
            "session_id": sid,
            "trajectory": trajectory,
            "product_categories": cats,
            "price_bucket": buckets,
            "intent": intent,
        }
        if persona_id_by_session is not None:
            record["persona_id"] = persona_id_by_session.get(str(sid), "")
            record["persona"] = ""
        records.append(record)

    records_df = pd.DataFrame(records)

    # All records have a non-empty generated intent. Keep this defensive filter
    # in case a caller supplies a custom intent strategy later.
    initial = len(records_df)
    records_df = records_df[records_df["intent"].str.strip() != ""]
    print(f"Prepared sessions: {initial} → {len(records_df)} (non-empty intent)")

    records_df.to_json(output_path, orient="records", lines=True)
    print(f"Wrote {len(records_df)} session records to {output_path}")


def enrich_sessions(
    df: pd.DataFrame,
    output_path: str,
    *,
    features_path: str,
    persona_id_by_session: dict[str, str] | None = None,
    persona_count: int | None = None,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    base_url: str = DEFAULT_BASE_URL,
    concurrency: int = 16,
    request_timeout: float = 600,
    overwrite_personas: bool = False,
) -> None:
    """Build session intents and generate personas from required buyer features."""
    if persona_id_by_session is None and "persona" in df.columns:
        persona_id_by_session = {
            str(session_id): str(group["persona"].dropna().iloc[0])
            for session_id, group in df.groupby("session_id", sort=False)
            if not group["persona"].dropna().empty
            and str(group["persona"].dropna().iloc[0]).strip()
        }

    to_sessions_jsonl(
        df,
        output_path,
        persona_id_by_session=persona_id_by_session,
    )

    print("Generating personas from buyer features...")
    generate_personas(
        features=features_path,
        sessions=output_path,
        n=persona_count,
        model=model,
        provider=provider,
        base_url=base_url,
        concurrency=concurrency,
        request_timeout=request_timeout,
        overwrite=overwrite_personas,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate session trajectories, intents, and personas"
    )
    parser.add_argument(
        "--input", default="data/sessions_external.csv", help="clickstream CSV"
    )
    parser.add_argument(
        "--csv-out",
        default="data/anonymized_clickstream.csv",
        help="Output anonymized CSV path",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/sessions.jsonl",
        help="Output sessions JSONL path",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Required buyer-feature JSON used to generate personas after intents",
    )
    parser.add_argument(
        "--persona-count",
        type=int,
        help="Number of buyers to enrich with personas (default: every buyer)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--overwrite-personas", action="store_true")
    args = parser.parse_args()

    print("Step 1: Read real session data...")
    df = pd.read_csv(args.input)

    print("\nStep 2: Writing anonymized CSV...")
    to_anonymized_csv(df, args.csv_out)

    print("\nStep 3: Enriching sessions with intents and personas...")
    enrich_sessions(
        df,
        args.jsonl_out,
        features_path=args.features,
        persona_count=args.persona_count,
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        overwrite_personas=args.overwrite_personas,
    )

if __name__ == "__main__":
    main()
