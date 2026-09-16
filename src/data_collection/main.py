"""Run the complete data-collection pipeline through one entry point.

Select a source processor, pass its provider-specific settings through
``meta_arg``, write the standardized collection artifacts, and enrich them into
``sessions.jsonl``.

Examples:
    python -m src.data_collection.main --processor clarity \
        --catalog products.csv --start 2026-03-01 --end 2026-03-31 \
        --output-dir output

    python -m src.data_collection.main --processor posthog \
        --catalog products.csv --project-id 386541 \
        --start 2026-04-15 --end 2026-04-17 --output-dir output

    python -m src.data_collection.main --processor bigquery \
        --project analytics-project \
        --table analytics-project.dataset.buyer_events_sessions \
        --shop-id 12345 --sample-sessions 900 --output-dir output
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .bigquery_processor import BigQueryProcessor
from .catalog import load_catalog
from .clarity_processor import ClarityProcessor
from .enrich_sessions import enrich_sessions
from .generate_personas import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_PROVIDER
from .posthog_processor import DEFAULT_HOST, DEFAULT_PROJECT_ID, PosthogProcessor
from .processor import Processor

PROCESSOR_TYPES = {
    "clarity": ClarityProcessor,
    "posthog": PosthogProcessor,
    "bigquery": BigQueryProcessor,
}


def _parse_date(value: str | None, *, default: datetime) -> datetime:
    if value is None:
        return default
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    end = _parse_date(args.end, default=now)
    start = _parse_date(args.start, default=end - timedelta(days=3))
    if start >= end:
        raise ValueError("--start must be earlier than --end")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processor",
        required=True,
        choices=sorted(PROCESSOR_TYPES),
        help="Collection source to run before enrichment",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for all collection and enrichment outputs",
    )
    parser.add_argument(
        "--catalog",
        help="Required products.csv for Clarity/PostHog; BigQuery builds it",
    )

    dates = parser.add_argument_group("Clarity and PostHog")
    dates.add_argument("--start", help="Start date (YYYY-MM-DD); default: 3 days ago")
    dates.add_argument("--end", help="End date (YYYY-MM-DD); default: now")

    clarity = parser.add_argument_group("Clarity")
    clarity.add_argument(
        "--window-days",
        type=int,
        default=2,
        help="Days per recordings request (default: 2; maximum: 3)",
    )
    clarity.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between recordings requests in seconds",
    )

    posthog = parser.add_argument_group("PostHog")
    posthog.add_argument("--host", default=DEFAULT_HOST)
    posthog.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    posthog.add_argument(
        "--limit", type=int, default=50_000, help="Maximum events per query"
    )

    bigquery = parser.add_argument_group("BigQuery")
    bigquery.add_argument("--project", help="GCP project for the BigQuery client")
    bigquery.add_argument("--table", help="Fully-qualified session table")
    bigquery.add_argument(
        "--shop-id", help="Optional source-specific website/shop identifier"
    )
    bigquery.add_argument(
        "--input-csv", help="Read a flat event export instead of querying BigQuery"
    )
    bigquery.add_argument(
        "--save-sql",
        nargs="?",
        const="src/data_collection/sql/buyer_events_to_clickstream.rendered.sql",
        help="Render BigQuery SQL to an optional path and exit",
    )
    bigquery.add_argument("--sample-sessions", type=int)
    bigquery.add_argument("--candidate-cap", type=int, default=50_000)
    bigquery.add_argument("--refresh-shop-products", action="store_true")
    bigquery.add_argument("--min-actions", type=int, default=3)
    bigquery.add_argument("--max-actions", type=int, default=20)
    bigquery.add_argument("--seed", type=int, default=42)
    bigquery.add_argument(
        "--explore-stay-mode",
        choices=["r1-only", "r1-gap"],
        default="r1-gap",
    )

    enrichment = parser.add_argument_group("Enrichment")
    enrichment.add_argument(
        "--persona-count",
        type=int,
        help="Number of buyers to generate personas for; default: every buyer",
    )
    enrichment.add_argument("--model", default=DEFAULT_MODEL)
    enrichment.add_argument("--provider", default=DEFAULT_PROVIDER)
    enrichment.add_argument("--base-url", default=DEFAULT_BASE_URL)
    enrichment.add_argument("--concurrency", type=int, default=16)
    enrichment.add_argument("--request-timeout", type=float, default=600)
    enrichment.add_argument("--overwrite-personas", action="store_true")
    return parser


def _load_required_catalog(
    args: argparse.Namespace, parser: argparse.ArgumentParser
):
    if not args.catalog:
        parser.error(f"--catalog is required for --processor {args.processor}")
    try:
        return load_catalog(args.catalog)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


def create_processor(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Processor:
    output_dir = Path(args.output_dir)

    if args.processor == "clarity":
        if not 1 <= args.window_days <= 3:
            parser.error("--window-days must be between 1 and 3")
        try:
            start, end = _date_range(args)
        except ValueError as exc:
            parser.error(str(exc))
        meta_arg = {
            "start": start,
            "end": end,
            "window_days": args.window_days,
            "delay": args.delay,
            "debug_dir": output_dir / "debug",
        }
        catalog = _load_required_catalog(args, parser)
        try:
            return ClarityProcessor(catalog=catalog, meta_arg=meta_arg)
        except ValueError as exc:
            parser.error(str(exc))

    if args.processor == "posthog":
        try:
            start, end = _date_range(args)
        except ValueError as exc:
            parser.error(str(exc))
        meta_arg = {
            "start": start,
            "end": end,
            "host": args.host,
            "project_id": args.project_id,
            "limit": args.limit,
        }
        catalog = _load_required_catalog(args, parser)
        try:
            return PosthogProcessor(catalog=catalog, meta_arg=meta_arg)
        except ValueError as exc:
            parser.error(str(exc))

    if not args.input_csv and not args.table:
        parser.error("--table is required unless --input-csv is provided")
    if args.min_actions > args.max_actions:
        parser.error("--min-actions cannot exceed --max-actions")
    meta_arg = {
        "project": args.project,
        "table": args.table,
        "shop_id": args.shop_id,
        "input_csv": args.input_csv,
        "sample_sessions": args.sample_sessions,
        "candidate_cap": args.candidate_cap,
        "refresh_shop_products": args.refresh_shop_products,
        "min_actions": args.min_actions,
        "max_actions": args.max_actions,
        "seed": args.seed,
        "explore_stay_mode": args.explore_stay_mode,
    }
    return BigQueryProcessor(catalog=None, meta_arg=meta_arg)


def _render_bigquery_sql(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.processor != "bigquery":
        parser.error("--save-sql is only valid for --processor bigquery")
    if not args.table:
        parser.error("--table is required with --save-sql")
    from .bigquery_processor import _render_sql

    output_path = Path(args.save_sql)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_sql(args.table, args.shop_id), encoding="utf-8")
    print(f"Wrote rendered SQL to {output_path}")


def run_pipeline(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    output_dir = Path(args.output_dir)
    processor = create_processor(args, parser)

    print(f"[pipeline] Collecting with {processor.source_name} ...")
    rows = processor.run(output_dir)
    if not rows:
        raise RuntimeError(f"{processor.source_name} produced no clickstream rows")

    features_path = output_dir / "buyers_feature.json"
    if not features_path.exists():
        raise RuntimeError(f"processor did not produce required {features_path}")

    print("[pipeline] Enriching sessions with intents and personas ...")
    external = pd.DataFrame(rows)[processor.external_fields]
    sessions_path = output_dir / "sessions.jsonl"
    enrich_sessions(
        external,
        str(sessions_path),
        features_path=str(features_path),
        persona_id_by_session=processor.persona_id_by_session(rows),
        persona_count=args.persona_count,
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        overwrite_personas=args.overwrite_personas,
    )
    print(f"[pipeline] Done: {sessions_path}")
    return sessions_path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.save_sql is not None:
        _render_bigquery_sql(args, parser)
        return
    try:
        run_pipeline(args, parser)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
