"""
PosthogProcessor
================
Standardize + anonymize PostHog events into the common ``semantic_action``
clickstream (see :class:`Processor`).

PostHog exposes a flat event stream (``$pageview`` / ``$pageleave`` /
``$autocapture``) via its HogQL API. This processor groups events by session,
splits a session whenever the landing ``?persona=`` changes, infers dwell from
the gap to the next pageview, and reads click text from ``elements_chain``.

Use :mod:`src.data_collection.main` to run collection and enrichment together.
"""

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

try:
    from .catalog import ProductCatalog
    from .processor import EXPLORE_STAY_THRESHOLD_MS, Processor
except ImportError:  # direct-file execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from catalog import ProductCatalog  # type: ignore[no-redef]
    from processor import (  # type: ignore[no-redef]
        EXPLORE_STAY_THRESHOLD_MS,
        Processor,
    )

DEFAULT_HOST = "https://us.posthog.com"
DEFAULT_PROJECT_ID = "386541"


class PostHogClient:
    """Minimal client for PostHog's HogQL query endpoint."""

    def __init__(self, host: str, project_id: str, api_key: str):
        self.host = host.rstrip("/")
        self.project_id = project_id
        self.api_key = api_key

    def query(self, hogql: str) -> tuple[list[list], list[str]]:
        response = requests.post(
            f"{self.host}/api/projects/{self.project_id}/query/",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"query": {"kind": "HogQLQuery", "query": hogql}},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", []), data.get("columns", [])

    def query_dicts(self, hogql: str) -> list[dict]:
        rows, columns = self.query(hogql)
        return [dict(zip(columns, row)) for row in rows]

    def player_link(self, session_id: str) -> str:
        return f"{self.host}/project/{self.project_id}/replay/{session_id}"


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _extract_click_text(elements_chain: str) -> str:
    match = re.search(r'text="([^"]*)"', elements_chain or "")
    return (match.group(1) if match else "").strip()


class PosthogProcessor(Processor):
    output_prefix = ""
    source_name = "posthog"

    def __init__(
        self,
        catalog: ProductCatalog,
        meta_arg: Mapping[str, Any] | None = None,
    ):
        super().__init__(catalog, meta_arg)
        api_key = self.get_meta_arg("api_key") or os.environ.get("POSTHOG_API_KEY")
        if not api_key:
            raise ValueError("set POSTHOG_API_KEY or provide meta_arg['api_key']")
        self.project_id = str(self.get_meta_arg("project_id", DEFAULT_PROJECT_ID))
        self.start = self.get_meta_arg("start")
        self.end = self.get_meta_arg("end")
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise ValueError(
                "PostHog requires datetime meta_arg values for start and end"
            )
        self.limit = self.get_meta_arg("limit", 50_000)
        self.client = PostHogClient(
            self.get_meta_arg("host", DEFAULT_HOST),
            self.project_id,
            api_key,
        )

    @staticmethod
    def _parse_ts(iso: str) -> int:
        try:
            dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError, AttributeError):
            return 0

    # -- PostHog API --------------------------------------------------------
    def fetch(self) -> list[dict]:
        query = f"""
SELECT event, timestamp, distinct_id,
       properties.$session_id AS session_id,
       properties.$current_url AS current_url,
       properties.$pathname AS pathname,
       properties.$referrer AS referrer,
       properties.persona AS persona,
       elements_chain
FROM events
WHERE timestamp >= toDateTime('{_format_datetime(self.start)}')
  AND timestamp <= toDateTime('{_format_datetime(self.end)}')
  AND event IN ('$pageview', '$pageleave', '$autocapture')
ORDER BY timestamp ASC
LIMIT {self.limit}
        """.strip()
        print(f"[posthog] fetching events {self.start.date()} -> {self.end.date()}")
        events = self.client.query_dicts(query)
        print(f"  got {len(events)} events")
        return events

    # -- transform ----------------------------------------------------------
    def _split_by_persona(self, evts: list[dict]) -> list[tuple[str, list[dict]]]:
        """Chunk a session's events wherever the landing ``?persona=`` changes."""
        chunks: list[tuple[str, list[dict]]] = []
        current_persona = None
        current: list[dict] = []
        for e in evts:
            landing = self.extract_persona(e.get("current_url") or "")
            if landing and landing != current_persona:
                if current:
                    chunks.append((current_persona or "", current))
                current_persona = landing
                current = []
            current.append(e)
        if current:
            chunks.append((current_persona or "", current))
        return chunks

    def _process_pageviews(self, evts, ids, persona) -> list[dict]:
        rows: list[dict] = []
        visited: list[str] = []
        pageviews = [e for e in evts if e.get("event") == "$pageview"]
        for idx, pv in enumerate(pageviews):
            url = pv.get("current_url") or ""
            pv_ts = self._parse_ts(pv.get("timestamp"))
            next_ts = (
                self._parse_ts(pageviews[idx + 1].get("timestamp"))
                if idx + 1 < len(pageviews)
                else pv_ts
            )
            duration_ms = max(0, next_ts - pv_ts)
            action, slug = self.infer_pageview_action(
                url, pv.get("referrer") or "", visited
            )

            primary_ts = pv_ts
            if duration_ms > EXPLORE_STAY_THRESHOLD_MS:
                rows.append(
                    self.make_row(ids, "explore-stay", pv_ts, url, slug, persona)
                )
                primary_ts = pv_ts + duration_ms
            rows.append(self.make_row(ids, action, primary_ts, url, slug, persona))
            visited.append(url)
        return rows

    def _process_clicks(self, evts, ids, persona) -> list[dict]:
        rows: list[dict] = []
        for e in evts:
            if e.get("event") != "$autocapture":
                continue
            click_text = _extract_click_text(e.get("elements_chain") or "")
            action = self.classify_click(click_text)
            if not action:
                continue
            url = e.get("current_url") or ""
            slug = self.extract_product_slug(url) or self.extract_product_slug(
                e.get("referrer") or ""
            )
            rows.append(
                self.make_row(
                    ids,
                    action,
                    self._parse_ts(e.get("timestamp")),
                    url,
                    slug,
                    persona,
                    None,
                    click_text,
                )
            )
        return rows

    def transform_to_clickstream(self, events: list[dict]) -> list[dict]:
        by_session: dict[str, list[dict]] = {}
        for e in events:
            sid = e.get("session_id") or ""
            if sid:
                by_session.setdefault(sid, []).append(e)

        store_uuid = self.deterministic_uuid(self.project_id)
        all_rows: list[dict] = []
        for sid, evts in by_session.items():
            evts.sort(key=lambda e: e.get("timestamp") or "")
            chunks = self._split_by_persona(evts)
            for chunk_idx, (chunk_persona, chunk) in enumerate(chunks):
                synth_sid = (
                    sid
                    if len(chunks) == 1
                    else f"{sid}::p{chunk_persona or 'none'}::{chunk_idx}"
                )
                persona = chunk_persona or next(
                    (
                        self.extract_persona(e.get("current_url") or "")
                        for e in chunk
                        if self.extract_persona(e.get("current_url") or "")
                    ),
                    "",
                )
                ids = (
                    self.deterministic_uuid(synth_sid),
                    self.deterministic_uuid(chunk[0].get("distinct_id") or synth_sid),
                    store_uuid,
                )
                session_rows = self._process_pageviews(
                    chunk, ids, persona
                ) + self._process_clicks(chunk, ids, persona)
                if not session_rows:
                    continue
                last_url = chunk[-1].get("current_url") or ""
                term_ts = max(r["timestamp"] for r in session_rows) + 1
                session_rows.append(
                    self.make_row(ids, "terminate", term_ts, last_url, "", persona)
                )
                session_rows.sort(key=lambda r: r["timestamp"])
                all_rows.extend(session_rows)
        return all_rows
