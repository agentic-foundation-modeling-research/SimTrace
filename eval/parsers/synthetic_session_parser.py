import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from eval.models import NormalizedSession
from .base_parser import BaseParser


class SyntheticSessionParser(BaseParser):
    """Parses synthetic session JSON from the LLM-based buyer simulator.

    Expected JSON structure: list of dicts with session_id, timestamp,
    synthetic_action (JSON string), clicked_url, url fields.

    When a product catalog is supplied, each step's ``product_texts`` entry
    is a formatted catalog string keyed off the product handle extracted from
    the step's ``clicked_url``.
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
                for key, _ in SyntheticSessionParser._CATALOG_FIELDS
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

    def parse(self, filepath: str) -> list[NormalizedSession]:
        with open(filepath, "r") as f:
            records = json.load(f)

        grouped = self._group_by_session(records, key="session_id")
        sessions = []

        for sid, events in grouped.items():
            events = self._sort_by_timestamp(events, key="timestamp")
            # # since in the LLM's search action, it will first click the search icon and then type the query,
            # # we remove the click event if it's immediately followed by a search type event to avoid double-counting search actions
            # events = self._remove_redundant_search_clicks(events)
            actions = []
            categories = []
            product_texts = []
            timestamps = []
            urls = []
            search_queries = []

            prev_url = ""
            prev_clicked_url = ""
            for event in events:
                parsed_action = self._parse_synthetic_action(event)
                # The terminate event carries no navigation; skip it here so it
                # isn't misclassified (e.g. as explore-goto). A single canonical
                # terminate step is appended after the loop.
                if (parsed_action.get("action") or "").lower() == "terminate":
                    continue
                semantic = self._classify_action(parsed_action, event, prev_url, prev_clicked_url)
                clicked_url = event.get("clicked_url", "")
                current_url = event.get("url", "")

                actions.append(semantic)
                handle = self._extract_handle(clicked_url) or self._extract_handle(current_url)
                product_texts.append(
                    self._format_product_text(handle)
                )
                categories.append(self._extract_category(handle))
                timestamps.append(event.get("timestamp", ""))
                urls.append(clicked_url)
                # The query is the typed text, captured only for actual search submissions.
                query = ""
                if semantic == "explore-search" and (parsed_action.get("action") or "").lower() == "type":
                    query = (parsed_action.get("text") or "").strip()
                search_queries.append(query)
                prev_url = current_url
                prev_clicked_url = clicked_url

            # Every session ends with terminate
            actions.append("terminate")
            categories.append("")
            product_texts.append("")
            timestamps.append(timestamps[-1] if timestamps else "")
            urls.append("")
            search_queries.append("")

            # Collapse redundant steps (consecutive explore-stay, redundant search steps).
            (actions, categories, product_texts, timestamps, urls, search_queries) = (
                self._deduplicate(
                    actions, categories, product_texts, timestamps, urls, search_queries
                )
            )

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

    @staticmethod
    def _parse_synthetic_action(event: dict) -> dict:
        raw = event.get("synthetic_action", "{}")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _deduplicate(actions, categories, product_texts, timestamps, urls, search_queries):
        """Post-process the classified step sequence, collapsing redundant steps.

        Rule A (explore-stay): a run of consecutive explore-stay collapses to its
            first step.
        Rule B (explore-search): within a run of consecutive explore-search, keep
            only the steps carrying a typed query (the real submissions); all
            distinct query steps are kept (including repeated text). If the run has
            no typed query, keep the first step and label its query "Failure".

        Operates on parallel lists via a keep-index pass so every list stays aligned.
        """
        keep: list[int] = []
        n = len(actions)
        i = 0
        while i < n:
            if actions[i] == "explore-stay":
                j = i
                while j < n and actions[j] == "explore-stay":
                    j += 1
                keep.append(i)  # first of the run
                i = j
            elif actions[i] == "explore-search":
                j = i
                while j < n and actions[j] == "explore-search":
                    j += 1
                query_idxs = [k for k in range(i, j) if search_queries[k]]
                if query_idxs:
                    keep.extend(query_idxs)
                else:
                    search_queries[i] = "Failure"
                    keep.append(i)  # one representative for a no-query run
                i = j
            else:
                keep.append(i)
                i += 1

        pick = lambda lst: [lst[k] for k in keep]
        return (
            pick(actions),
            pick(categories),
            pick(product_texts),
            pick(timestamps),
            pick(urls),
            pick(search_queries),
        )

    @staticmethod
    def _remove_redundant_search_clicks(events: list[dict]) -> list[dict]:
        result = []
        for i, event in enumerate(events):
            parsed = SyntheticSessionParser._parse_synthetic_action(event)
            action = (parsed.get("action") or "").lower()
            target = (parsed.get("target") or "").lower()
            if action == "click" and "search" in target:
                if i + 1 < len(events):
                    next_parsed = SyntheticSessionParser._parse_synthetic_action(
                        events[i + 1]
                    )
                    next_action = (next_parsed.get("action") or "").lower()
                    next_target = (next_parsed.get("target") or "").lower()
                    if next_action == "type" and "search" in next_target:
                        continue
            result.append(event)
        return result

    @staticmethod
    def _classify_action(parsed_action: dict, event: dict, prev_url: str = "", prev_clicked_url: str = "") -> str:
        """Map synthetic action + URL patterns to a semantic action label.

        Uses clicked_url, the next page's url, target name, and description
        to classify actions. The LLM may click buttons (e.g., "Buy it now",
        "Add to cart") that don't produce a clicked_url but do change the
        page URL or have recognizable target/description text.
        """
        action_type = (parsed_action.get("action") or "").lower()
        clicked_url = (event.get("clicked_url") or "").lower()
        current_url = (event.get("url") or "").lower()
        target = (parsed_action.get("target") or "").lower()
        description = (parsed_action.get("description") or "").lower()

        if action_type == "back":
            return "explore-back"

        if action_type == "scroll" or action_type == "hover" or action_type == "move":
            return "explore-stay"

        # Check clicked_url patterns first
        if "/products/" in clicked_url:
            return "detail"

        if "/search" in clicked_url or "?q=" in clicked_url or "search" in target:
            return "explore-search"

        if "/collections/" in clicked_url:
            return "explore-goto"

        if "/cart" in clicked_url:
            return "add"

        if "/checkout" in clicked_url or "/checkouts/" in clicked_url:
            return "checkout"

        # Check target and description for add/checkout signals
        # (buttons like "Buy it now" or "Add to cart" have empty clicked_url)
        add_signals = ["add_to_cart", "add_to_bag", "add_item"]
        checkout_signals = ["buy_it_now", "buy_now", "checkout", "check_out"]

        if any(s in target for s in add_signals):
            return "add"
        if any(s in target for s in checkout_signals):
            return "checkout"
        if any(
            phrase in description
            for phrase in ["add to cart", "add to bag", "add item"]
        ):
            return "add"
        if any(
            phrase in description
            for phrase in ["buy it now", "buy now", "proceed to checkout", "checkout"]
        ):
            return "checkout"

        if action_type == "type":
            current_url = event.get("url", "").lower()
            if "/search" in current_url or "?q=" in current_url:
                return "explore-search"

        if (current_url and current_url == prev_url.lower()) or \
            (current_url.split("?")[0].endswith(prev_clicked_url.lower())) or \
            (prev_url.split("?")[0] == current_url.split("?")[0]):
            return "explore-stay"

        # Default: any navigation is explore-goto
        return "explore-goto"

    def _extract_category(self, handle: str) -> str:
        handle = str(handle).strip()
        if not handle:
            return ""
        fields = self._catalog.get(handle)
        if not fields:
            return ""
        return fields.get("product_type", "")
        

    @staticmethod
    def _extract_handle(url: str) -> str:
        """Extract the product handle from a /products/<handle> URL."""
        path = urlparse(url).path.lower().strip("/")

        if "/products/" in f"/{path}":
            parts = path.split("products/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]

        return ""
