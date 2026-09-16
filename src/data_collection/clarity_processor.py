"""
ClarityProcessor
================
Standardize + anonymize Microsoft Clarity session recordings into the common
``semantic_action`` clickstream (see :class:`Processor`).

Clarity exposes recordings through its MCP API as a per-session ``timeline`` of
pages, each with ``timelineEvents`` (clicks) and a dwell ``duration``. This
processor windows over the (max 3-day) lookback, reconstructs event timestamps
from page start-offsets, and maps each page onto a semantic action.

Use :mod:`src.data_collection.main` to run collection and enrichment together.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
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

MCP_BASE = "https://clarity.microsoft.com/mcp"
RECORDINGS_URL = f"{MCP_BASE}/recordings/sample"
DASHBOARD_URL = f"{MCP_BASE}/dashboard/query"
MAX_RECORDINGS_PER_REQUEST = 250
SORT_SESSION_START_DESC = 0


class ClarityProcessor(Processor):
    output_prefix = ""  # sessions_raw.csv / sessions_external.csv
    source_name = "clarity"

    def __init__(
        self,
        catalog: ProductCatalog,
        meta_arg: Mapping[str, Any] | None = None,
    ):
        super().__init__(catalog, meta_arg)
        self.token = self.get_meta_arg("token") or os.environ.get("CLARITY_API_TOKEN")
        if not self.token:
            raise ValueError("set CLARITY_API_TOKEN or provide meta_arg['token']")
        self.start = self.get_meta_arg("start")
        self.end = self.get_meta_arg("end")
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise ValueError("Clarity requires datetime meta_arg values for start and end")
        self.window_days = self.get_meta_arg("window_days", 2)
        self.delay = self.get_meta_arg("delay", 1.0)
        debug_dir = self.get_meta_arg("debug_dir")
        self.debug_dir = Path(debug_dir) if debug_dir else None

    # -- Clarity API --------------------------------------------------------
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _iso_utc(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def fetch_recordings(self, start: datetime, end: datetime, count: int) -> list:
        # The date filter MUST live inside `filters`, or the API returns 500.
        body = {
            "sortBy": SORT_SESSION_START_DESC,
            "start": self._iso_utc(start),
            "end": self._iso_utc(end),
            "count": count,
            "filters": {
                "date": {"start": self._iso_utc(start), "end": self._iso_utc(end)}
            },
        }
        resp = requests.post(
            RECORDINGS_URL, headers=self._headers(), json=body, timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_dashboard(self, query: str, timezone_str: str = "UTC") -> dict:
        resp = requests.post(
            DASHBOARD_URL,
            headers=self._headers(),
            json={"query": query, "timezone": timezone_str},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch(self) -> list[dict]:
        """Window over the date range (Clarity max lookback 3 days) and dedupe."""
        print(
            f"[clarity] extracting recordings {self.start.date()} -> {self.end.date()}"
        )
        all_sessions: list[dict] = []
        seen: set[str] = set()
        current_end = self.end
        req = 0
        while current_end > self.start:
            current_start = max(
                self.start, current_end - timedelta(days=self.window_days)
            )
            req += 1
            print(
                f"  request #{req}: {current_start.date()} -> {current_end.date()}",
                end="",
            )
            try:
                raw = self.fetch_recordings(
                    current_start, current_end, MAX_RECORDINGS_PER_REQUEST
                )
                sessions = raw if isinstance(raw, list) else []
                new = []
                for s in sessions:
                    sid = (
                        s.get("link")
                        or s.get("session_id")
                        or s.get("id")
                        or json.dumps(s, sort_keys=True)
                    )
                    if sid not in seen:
                        seen.add(sid)
                        new.append(s)
                all_sessions.extend(new)
                print(f" -> {len(new)} new (total {len(all_sessions)})")
            except requests.HTTPError as e:
                print(f" -> ERROR {e.response.status_code}: {e.response.text[:200]}")
                if e.response.status_code == 429:
                    print("  rate limited; stopping.")
                    break
            current_end = current_start
            if current_end > self.start:
                time.sleep(self.delay)
        print(f"  total unique sessions: {len(all_sessions)}")
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "raw_sessions.json").write_text(
                json.dumps(all_sessions, indent=2, default=str)
            )
        return all_sessions

    # -- Clarity-specific parsing -------------------------------------------
    @staticmethod
    def _parse_session_link(link: str) -> tuple[str, str, str]:
        """(project_id, user_id, session_id) from a Clarity player link."""
        parts = link.rstrip("/").split("/")
        if len(parts) >= 3:
            return parts[-3], parts[-2], parts[-1]
        return "", "", link

    @staticmethod
    def _parse_timestamp(ts_str: str) -> int:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    @staticmethod
    def _parse_start_offset(start_str: str) -> int:
        """Timeline 'start' like '00:04' or '01:23:45' -> milliseconds."""
        parts = start_str.split(":")
        try:
            if len(parts) == 2:
                return (int(parts[0]) * 60 + int(parts[1])) * 1000
            if len(parts) == 3:
                return (
                    int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                ) * 1000
        except ValueError:
            pass
        return 0

    @classmethod
    def _classify_click_events(cls, timeline_events: list) -> list[dict]:
        """Classify Clarity click ``timelineEvents`` into add/checkout/remove."""
        classified = []
        for evt in timeline_events:
            if not isinstance(evt, dict) or evt.get("eventtype", "").lower() != "click":
                continue
            action = cls.classify_click(evt.get("text") or "")
            if action:
                classified.append(
                    {
                        "semantic_action": action,
                        "start": evt.get("start", "00:00"),
                        "text": evt.get("text", ""),
                    }
                )
        return classified

    # -- transform ----------------------------------------------------------
    def transform_to_clickstream(self, sessions: list[dict]) -> list[dict]:
        print("[clarity] transforming to clickstream...")
        rows: list[dict] = []
        for session in sessions:
            rows.extend(self._transform_session(session))
        print(f"  generated {len(rows)} events from {len(sessions)} sessions")
        return rows

    def _transform_session(self, session: dict) -> list[dict]:
        link = session.get("link", "")
        project_id, user_id, session_id = self._parse_session_link(link)
        base_ts = self._parse_timestamp(session.get("timestamp", ""))
        ids = (
            self.deterministic_uuid(session_id or link),
            self.deterministic_uuid(user_id) if user_id else "",
            self.deterministic_uuid(project_id),
        )
        timeline = session.get("timeline", [])

        persona = ""
        for page in timeline:
            persona = self.extract_persona(page.get("url", ""))
            if persona:
                break

        session_rows: list[dict] = []
        visited: list[str] = []
        for page in timeline:
            page_url = page.get("url") or ""
            referrer = page.get("referrerUrl") or ""
            duration_ms = page.get("duration", 0)
            start_offset_ms = self._parse_start_offset(page.get("start", "00:00"))
            action, slug = self.infer_pageview_action(page_url, referrer, visited)
            created_at = datetime.fromtimestamp(
                (base_ts + start_offset_ms) / 1000, tz=timezone.utc
            ).isoformat()

            stayed = duration_ms > EXPLORE_STAY_THRESHOLD_MS
            if stayed:
                session_rows.append(
                    self.make_row(
                        ids,
                        "explore-stay",
                        base_ts + start_offset_ms,
                        page_url,
                        slug,
                        persona,
                        created_at,
                    )
                )
            primary_ts = base_ts + start_offset_ms + (duration_ms if stayed else 0)
            session_rows.append(
                self.make_row(
                    ids,
                    action,
                    primary_ts,
                    page_url,
                    slug,
                    persona,
                    created_at,
                )
            )

            for ca in self._classify_click_events(page.get("timelineEvents", [])):
                ca_slug = slug or self.extract_product_slug(referrer)
                session_rows.append(
                    self.make_row(
                        ids,
                        ca["semantic_action"],
                        base_ts + self._parse_start_offset(ca["start"]),
                        page_url,
                        ca_slug,
                        persona,
                        created_at,
                        ca["text"],
                    )
                )
            visited.append(page_url)

        if timeline:
            last = timeline[-1]
            page_end_ts = (
                base_ts
                + self._parse_start_offset(last.get("start", "00:00"))
                + last.get("duration", 0)
            )
            term_ts = (
                max(
                    page_end_ts,
                    max((r["timestamp"] for r in session_rows), default=page_end_ts),
                )
                + 1
            )
            term_created = datetime.fromtimestamp(
                term_ts / 1000, tz=timezone.utc
            ).isoformat()
            session_rows.append(
                self.make_row(
                    ids,
                    "terminate",
                    term_ts,
                    last.get("url") or "",
                    "",
                    persona,
                    term_created,
                )
            )
        return session_rows

    # -- utilities ----------------------------------------------------------
    def check_data_availability(self) -> None:
        """Quick NL dashboard query for total session/user counts."""
        try:
            result = self.fetch_dashboard(
                "What is the total number of sessions and users recorded all time? "
                "Show the earliest and latest session dates."
            )
            print(json.dumps(result, indent=2))
        except Exception as e:  # noqa: BLE001 - diagnostic helper
            print(f"  error checking data: {e}")
