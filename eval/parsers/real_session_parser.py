import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from eval.models import NormalizedSession
from .base_parser import BaseParser


class RealSessionParser(BaseParser):
    """Parses clickstream CSV into NormalizedSession objects.

    Supports both the anonymized format (product_hash, url_hash) and
    the Clarity raw format (url with full URLs). Product slugs are
    extracted from URLs when a dedicated product_slug column is absent.

    When a product catalog is supplied, each step's ``product_texts`` entry
    is a formatted catalog string keyed off the session row's product handle.
    """

    # Ordered (catalog column -> display label) used to build product_texts.
    _CATALOG_FIELDS = [
        ("product_type", "Product Type"),
        ("product_title", "Product Title"),
        ("product_description", "Product Description"),
        ("product_handle", "Product Handle"),
    ]

    def __init__(self, product_catalog: str | None = None):
        self._catalog = self._load_catalog(product_catalog)

    @staticmethod
    def _load_catalog(path: str | None) -> dict[str, dict[str, str]]:
        """Load a product catalog CSV into ``{product_handle: {field: value}}``."""
        if not path or not Path(path).exists():
            return {}
        df = pd.read_csv(path, dtype=str).fillna("")
        catalog: dict[str, dict[str, str]] = {}
        for _, row in df.iterrows():
            handle = str(row.get("product_handle", "")).strip()
            if not handle:
                continue
            catalog[handle] = {
                key: str(row.get(key, "")).strip()
                for key, _ in RealSessionParser._CATALOG_FIELDS
            }
        return catalog

    def _format_product_text(self, handle: str) -> str:
        """Build the catalog string for a handle, omitting empty fields."""
        handle = str(handle).strip()
        if not handle:
            return ""
        fields = self._catalog.get(handle)
        if not fields:
            return ""
        parts = [
            f"{label}: {fields[key]}"
            for key, label in self._CATALOG_FIELDS
            if fields.get(key)
        ]
        return ", ".join(parts)

    @staticmethod
    def _extract_query(raw_action: str, semantic_action: str) -> str:
        """Pull the search query string out of a raw_action JSON payload.
        Only explore-search rows carry a query; everything else returns ""."""
        if semantic_action != "explore-search":
            return ""
        try:
            return (json.loads(raw_action) or {}).get("query", "") or ""
        except (json.JSONDecodeError, TypeError):
            return ""

    @staticmethod
    def _extract_slug(url: str) -> str:
        if not isinstance(url, str):
            return ""
        path = urlparse(url).path.lower().strip("/")
        if "/products/" in f"/{path}":
            parts = path.split("products/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]
        return ""

    def parse(self, filepath: str) -> list[NormalizedSession]:
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["session_id", "timestamp"])
        df = df.sort_values(["session_id", "timestamp"])
        
        has_url = "url" in df.columns
        has_raw_url = "url_raw" in df.columns

        sessions = []
        for sid, group in df.groupby("session_id", sort=False):
            group = group.sort_values("timestamp")
            actions = group["semantic_action"].fillna("").tolist()
            categories = (
                group.get("product_category", pd.Series(dtype=str)).fillna("").tolist()
            )
            if not categories:
                categories = [""] * len(actions)

            handle_col = (
                "product_handle"
                if "product_handle" in group.columns
                else ("product_slug" if "product_slug" in group.columns else None)
            )
            if handle_col:
                handles = group[handle_col].fillna("").tolist()
            else:
                handles = [""] * len(actions)
            product_texts = [self._format_product_text(h) for h in handles]

            timestamps = group["timestamp"].astype(str).tolist()

            url_col = "url" if has_url else ("url_raw" if has_raw_url else "url_hash")
            urls = (
                group[url_col].fillna("").tolist()
                if url_col in group.columns
                else [""] * len(actions)
            )

            if "raw_action" in group.columns:
                raw_actions = group["raw_action"].fillna("").tolist()
                search_queries = [
                    self._extract_query(raw, act)
                    for raw, act in zip(raw_actions, actions)
                ]
            else:
                search_queries = [""] * len(actions)

            session = NormalizedSession(
                session_id=str(sid),
                actions=actions,
                product_categories=categories,
                product_texts=product_texts,
                timestamps=timestamps,
                urls=urls,
                search_queries=search_queries,
            )

            if self._validate_session(session):
                sessions.append(session)

        return sessions
