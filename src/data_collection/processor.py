"""
Processor — base class for clickstream standardization + anonymization.
========================================================================
A ``Processor`` turns the raw session logs of one analytics provider into the
common, anonymized ``semantic_action`` clickstream used by the rest of the
pipeline (paper §2.1, "standardize and anonymize records").

The source-specific parts — authenticating, fetching, and mapping the
provider's event schema onto per-session rows — live in subclasses
(:class:`ClarityProcessor`, :class:`PosthogProcessor`). Everything that is the
same across providers lives here: source configuration, deterministic
hashing/anonymization, common commerce URL → semantic-action rules, price bucketing,
raw-vs-external column projection, and CSV writing.

To add a new source, subclass ``Processor`` and implement :meth:`fetch` and
:meth:`transform_to_clickstream`.
"""

import csv
import hashlib
import json
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

try:
    from .catalog import ProductCatalog
except ImportError:  # direct-file execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from catalog import ProductCatalog  # type: ignore[no-redef]

# A single page view longer than this is treated as scroll/browse and emitted as
# an explore-stay before the page's primary action (providers rarely log scroll).
EXPLORE_STAY_THRESHOLD_MS = 10_000

ADD_TO_CART_PATTERNS = ["add to cart", "add to bag", "add item"]
CHECKOUT_PATTERNS = [
    "check out",
    "checkout",
    "proceed to checkout",
    "buy now",
    "buy it now",
]
REMOVE_PATTERNS = ["remove", "delete", "remove item"]

# Full clickstream (one row per semantic action). Used as the eval `real_data`.
RAW_FIELDS = [
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
    "product_slug",
    "price",
]

# Anonymized, shareable projection: drops every free-text / plaintext field
# (raw_action, url, product_id, product_slug, price, created_at), keeping only
# hashed ids, bucketed price, and category. This projection IS the anonymization
# boundary — nothing downstream of `sessions_external.csv` sees raw content.
EXTERNAL_FIELDS = [
    "session_id",
    "user_id",
    "store_id",
    "timestamp",
    "semantic_action",
    "product_hash",
    "product_category",
    "price_bucket",
    "url_hash",
    "persona",
]


class Processor(ABC):
    """Standardize + anonymize one provider's sessions into the common schema."""

    #: Optional prefix for programmatic callers that need source-qualified CSVs.
    #: The unified pipeline leaves this empty to produce standard filenames.
    output_prefix: str = ""
    #: Short label used in log lines.
    source_name: str = "processor"
    #: Subclasses may extend either schema while reusing the shared writer.
    raw_fields: list[str] = RAW_FIELDS
    external_fields: list[str] = EXTERNAL_FIELDS

    def __init__(
        self,
        catalog: ProductCatalog | None,
        meta_arg: Mapping[str, Any] | None = None,
    ):
        # BigQuery defers construction until it has exported products.csv.
        # Every transformation path must still install a catalog before use.
        self.catalog = catalog
        # Keep source-only configuration out of the common constructor/API.
        # A copy prevents callers from mutating a running processor's settings.
        self.meta_arg = dict(meta_arg or {})

    def get_meta_arg(self, name: str, default: Any = None) -> Any:
        """Return one source-specific setting supplied through ``meta_arg``."""
        return self.meta_arg.get(name, default)

    # -- provider-specific seam ---------------------------------------------
    @abstractmethod
    def fetch(self) -> Any:
        """Authenticate and pull raw sessions/events from the provider API."""

    @abstractmethod
    def transform_to_clickstream(self, raw: Any) -> list[dict]:
        """Map the provider's raw records onto standardized clickstream rows."""

    # -- orchestration ------------------------------------------------------
    def run(self, output_dir: Path) -> list[dict]:
        """Fetch, transform, clean, and write clickstream plus buyer features."""
        self.require_catalog()
        raw = self.fetch()
        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
            print(f"[{self.source_name}] no data fetched; nothing to write.")
            return []
        rows = self.transform_to_clickstream(raw)
        rows = self.clean_noise_actions(rows)
        self.write_clickstream_csv(rows, output_dir)
        self.write_buyer_features(rows, output_dir)
        return rows

    def require_catalog(self) -> ProductCatalog:
        """Return the configured catalog or fail before producing partial data."""
        if self.catalog is None:
            raise ValueError(
                f"{self.source_name} requires a product catalog before processing"
            )
        return self.catalog

    # -- anonymization primitives (shared, paper §2.1) ----------------------
    @staticmethod
    def hash_value(value: str) -> str:
        """SHA-256 of a value, truncated to 16 hex chars (irreversible id/url hash)."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    @staticmethod
    def deterministic_uuid(value: str) -> str:
        """Stable pseudonymous UUID for a raw identifier (uuid5, NAMESPACE_URL)."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))

    @staticmethod
    def extract_persona(url: str) -> str:
        """Persona id from a ``?persona=`` query parameter (empty if absent)."""
        return parse_qs(urlparse(url or "").query).get("persona", [""])[0]
    # -- shared commerce-website URL semantics ------------------------------
    @staticmethod
    def extract_product_slug(url: str) -> str:
        """Slug from a conventional ``/products/<slug>`` URL, else empty string."""
        path = urlparse(url or "").path.rstrip("/").lower()
        if "/products/" in path:
            return path.split("/products/")[1].split("/")[0].split("?")[0]
        return ""

    @classmethod
    def classify_click(cls, click_text: str) -> str | None:
        """Map click/button text to add / checkout / remove, or None."""
        t = (click_text or "").lower().strip()
        if not t:
            return None
        if any(p in t for p in ADD_TO_CART_PATTERNS):
            return "add"
        if any(p in t for p in CHECKOUT_PATTERNS):
            return "checkout"
        if any(p in t for p in REMOVE_PATTERNS):
            return "remove"
        return None

    @classmethod
    def infer_pageview_action(
        cls, url: str, referrer: str, visited: list[str]
    ) -> tuple[str, str]:
        """Infer ``(semantic_action, product_slug)`` from common website URLs."""
        parsed = urlparse(url or "")
        path = parsed.path.rstrip("/").lower()
        query = parse_qs(parsed.query)
        ref_path = urlparse(referrer or "").path.rstrip("/").lower()
        slug = cls.extract_product_slug(url)

        if "/search" in path or "q" in query:
            return "explore-search", ""
        if "/products/" in path:
            return "detail", slug
        if "/checkout" in path or "/checkouts" in path:
            return "checkout", ""
        if path.endswith("/cart") or "/cart/" in path:
            if "/products/" in ref_path:
                return "add", cls.extract_product_slug(referrer)
            return "explore-goto", ""
        if url in visited:
            return "explore-back", ""
        return "explore-goto", ""

    @staticmethod
    def build_raw_action(action: str, url: str, slug: str, click_text: str = "") -> str:
        """JSON description of the original user action (kept only in the raw file)."""
        q = parse_qs(urlparse(url or "").query).get("q", [""])[0]
        payloads: dict[str, dict] = {
            "explore-search": {"query": q},
            "explore-back": {"url": url, "direction": "back"},
            "explore-stay": {"action": "scroll/browse"},
            "explore-goto": {"url": url},
            "detail": {"product": slug, "url": url},
            "add": ({"product": slug} if slug else {})
            | ({"button_text": click_text} if click_text else {"url": url}),
            "checkout": {"button_text": click_text} if click_text else {"url": url},
            "remove": {"button_text": click_text or "Remove"},
            "terminate": {"action": "session_end"},
        }
        return json.dumps(payloads.get(action, {"url": url}))

    def make_row(
        self,
        ids: tuple[str, str, str],
        action: str,
        ts: int,
        url: str,
        slug: str,
        persona: str = "",
        created_at: str | None = None,
        click_text: str = "",
    ) -> dict:
        """Build one standardized clickstream row (hashes ids/urls, buckets price)."""
        session_uuid, user_uuid, store_uuid = ids
        p_hash, p_id = (
            (self.hash_value(slug), self.deterministic_uuid(slug)) if slug else ("", "")
        )
        category, bucket, price = self.require_catalog().lookup(slug)
        if created_at is None:
            created_at = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        return {
            "session_id": session_uuid,
            "user_id": user_uuid,
            "store_id": store_uuid,
            "timestamp": ts,
            "semantic_action": action,
            "raw_action": self.build_raw_action(action, url, slug, click_text),
            "product_hash": p_hash,
            "product_id": p_id,
            "url_hash": self.hash_value(url),
            "url": url,
            "created_at": created_at,
            "persona": persona,
            "product_category": category,
            "price_bucket": bucket,
            "product_slug": slug,
            "price": price,
        }

    # -- cleaning + output --------------------------------------------------
    def clean_noise_actions(self, rows: list[dict]) -> list[dict]:
        """Trim pre-``?persona`` noise: drop each session's events up to and
        including the first row whose raw_action URL carries ``?persona``."""
        cutoffs: dict[str, int] = {}
        by_session: dict[str, list[dict]] = {}
        for row in rows:
            by_session.setdefault(row["session_id"], []).append(row)

        for session_id, session_rows in by_session.items():
            for row in sorted(session_rows, key=lambda r: r["timestamp"]):
                try:
                    raw = json.loads(row["raw_action"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(raw.get("url"), str) and "?persona" in raw["url"]:
                    cutoffs[session_id] = row["timestamp"]
                    break

        n_before = len({r["session_id"] for r in rows})
        cleaned = [
            row
            for row in rows
            if row["session_id"] not in cutoffs
            or row["timestamp"] > cutoffs[row["session_id"]]
        ]
        n_after = len({r["session_id"] for r in cleaned})
        avg_before = len(rows) / n_before if n_before else 0
        avg_after = len(cleaned) / n_after if n_after else 0
        print(
            f"[{self.source_name}] sessions: {n_before} → {n_after} | "
            f"avg rows/session: {avg_before:.2f} → {avg_after:.2f}"
        )
        return cleaned

    def write_clickstream_csv(self, rows: list[dict], output_dir: Path) -> None:
        """Write the raw (full) and external (anonymized) CSVs into ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = [
            (output_dir / f"{self.output_prefix}sessions_raw.csv", self.raw_fields),
            (
                output_dir / f"{self.output_prefix}sessions_external.csv",
                self.external_fields,
            ),
        ]
        print(f"[{self.source_name}] writing {len(rows)} rows...")
        for path, fields in targets:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  -> {path}")

    def write_buyer_features(self, rows: list[dict], output_dir: Path) -> Path:
        """Aggregate standardized rows and write ``buyers_feature.json``."""
        try:
            from .enrich_sessions import aggregate_clickstream_buyer_features
        except ImportError:  # direct-file execution fallback
            from enrich_sessions import aggregate_clickstream_buyer_features

        features = aggregate_clickstream_buyer_features(rows)
        output_path = Path(output_dir) / "buyers_feature.json"
        output_path.write_text(
            json.dumps(features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{self.source_name}] -> {output_path} ({len(features)} buyers)")
        return output_path

    def persona_id_by_session(self, rows: list[dict]) -> dict[str, str]:
        """Return the session-to-buyer join used by persona enrichment.

        Browser providers prefer an explicit ``?persona=`` value. Otherwise,
        use the same stable user/website key as the provider-neutral feature
        aggregation, falling back to the session id when no user is available.
        """
        by_session: dict[str, list[dict]] = {}
        for row in rows:
            by_session.setdefault(str(row.get("session_id") or ""), []).append(row)

        result: dict[str, str] = {}
        for session_id, session_rows in by_session.items():
            persona = next(
                (
                    str(row.get("persona"))
                    for row in session_rows
                    if str(row.get("persona") or "").strip()
                ),
                "",
            )
            if persona:
                result[session_id] = persona
                continue
            user_id = next(
                (
                    str(row.get("user_id"))
                    for row in session_rows
                    if str(row.get("user_id") or "").strip()
                ),
                "",
            )
            store_id = next(
                (
                    str(row.get("store_id"))
                    for row in session_rows
                    if str(row.get("store_id") or "").strip()
                ),
                "",
            )
            result[session_id] = (
                f"{user_id}-{store_id}" if user_id else session_id
            )
        return result
