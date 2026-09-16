"""Clickstream CSV parsing + trajectory model for ``clickstream_replay``.

Pure value layer for the
:class:`~shop_arena.gen.build.verifiers.clickstream_replay.ClickstreamReplayVerifier`.
Turns a real user **clickstream** CSV — ordered per-session actions on
the *live* store — into deterministic, sampled :class:`Session` objects
whose paths can be replayed against the generated **twin**.

The CSV is the export produced by the storefront analytics pipeline. Only
a handful of its columns are consumed:

* ``session_id`` — groups rows into a single user trajectory.
* ``timestamp`` — integer-milliseconds ordering key within a session.
* ``semantic_action`` — the high-level action to reproduce (``detail``,
  ``add``, ``explore-stay``, ``terminate``, …).
* ``raw_action`` — the raw analytics action blob (kept for feedback).
* ``url`` — the path the user was on (e.g. ``/products/<handle>``); the
  primary source of the replay target.
* ``product_handle`` — the product the row concerns, when applicable.
* ``store_id`` — used only when the caller filters to a single store.
* ``landing_url`` — full live URL; consulted **only** as a fallback
  source of the path when ``url`` is empty.

**URL mapping.** The verifier never touches the live store: the path
carried by a row (its ``url`` column, or the path component of
``landing_url``) is joined verbatim onto the twin's dev-server base URL.
In twin (``catalog_source="ingest"``) mode the twin preserves the live
store's handles and paths, so a live ``/products/blue-shirt`` maps to
``<base_url>/products/blue-shirt`` unchanged.

Module is import-safe: no I/O, no env reads, no side effects at import.
CSV reads happen only inside :func:`load_sessions`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

DEFAULT_MAX_SESSIONS: Final[int] = 15
"""Default cap on sampled sessions (mirrors the CLI/config default)."""

# Canonical ``semantic_action`` tokens the replayer knows how to reproduce.
ACTION_DETAIL: Final[str] = "detail"
"""View a product-detail page."""

ACTION_ADD: Final[str] = "add"
"""Add the row's product to the cart."""

ACTION_EXPLORE_STAY: Final[str] = "explore-stay"
"""Scroll / browse in place — no navigation."""

ACTION_TERMINATE: Final[str] = "terminate"
"""End the session — always trivially reproducible."""

_HTTP_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
"""URL schemes whose path component is a usable replay target."""


@dataclass(frozen=True, slots=True)
class ClickstreamEvent:
    """One ordered action within a user session.

    Attributes:
        session_id: Session this event belongs to.
        timestamp: Ordering key (integer milliseconds). Rows whose
            ``timestamp`` cannot be parsed sort as ``0`` but keep their
            original CSV order via a stable secondary key applied in
            :func:`load_sessions`.
        semantic_action: High-level action token (e.g. ``detail``).
            Normalised to lower-case, stripped.
        raw_action: Raw analytics action string, preserved verbatim for
            feedback rendering.
        url: The ``url`` column verbatim (a path such as
            ``/products/<handle>``, or empty).
        product_handle: The row's product handle, or empty.
        landing_url: The ``landing_url`` column verbatim; consulted only
            as a fallback path source when :attr:`url` is empty.
    """

    session_id: str
    timestamp: int
    semantic_action: str
    raw_action: str
    url: str
    product_handle: str
    landing_url: str

    def target_path(self) -> str | None:
        """Return the replay path for this event, or ``None``.

        Prefers the ``url`` column; falls back to the path component of
        ``landing_url``. A row that carries neither (e.g. a
        ``terminate`` marker) yields ``None`` — there is nothing to
        navigate to.

        Returns:
            A root-relative path (``/...``, optionally with a query
            string), or ``None`` when no path can be derived.
        """
        return _extract_path(self.url) or _extract_path(self.landing_url)

    def target_url(self, base_url: str) -> str | None:
        """Map this event's path onto the twin ``base_url``.

        Args:
            base_url: The twin dev-server base URL
                (``http://127.0.0.1:<port>``).

        Returns:
            ``base_url`` joined with :meth:`target_path`, or ``None``
            when the event has no path.
        """
        path = self.target_path()
        if path is None:
            return None
        return base_url.rstrip("/") + path


@dataclass(frozen=True, slots=True)
class Session:
    """A single user trajectory: an ordered tuple of events.

    Attributes:
        session_id: The session identifier.
        events: Events in replay order (ascending timestamp, stable on
            ties).
    """

    session_id: str
    events: tuple[ClickstreamEvent, ...]


@dataclass(frozen=True, slots=True)
class SampledSessions:
    """Result of parsing + sampling a clickstream CSV.

    Attributes:
        sessions: The deterministically sampled sessions (at most
            ``max_sessions``), ordered by ``session_id``.
        total_sessions: Number of distinct sessions matched *before*
            sampling (after any ``store_id`` filter).
        dropped: Sessions discarded by the cap
            (``total_sessions - len(sessions)``).
    """

    sessions: tuple[Session, ...]
    total_sessions: int
    dropped: int


def load_sessions(
    csv_path: Path,
    *,
    store_id: str | None = None,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
) -> SampledSessions:
    """Parse ``csv_path`` into deterministically sampled sessions.

    Rows are grouped by ``session_id``; each session's events are
    ordered by ``timestamp`` (integer milliseconds), ties broken by
    original CSV row order. Sessions are then sampled deterministically:
    session ids are sorted lexicographically and the first
    ``max_sessions`` are kept.

    Args:
        csv_path: Path to the clickstream CSV. Must exist and be
            readable.
        store_id: When set, only rows whose ``store_id`` column equals
            this value are retained.
        max_sessions: Upper bound on the number of sessions returned.
            Non-positive values disable the cap (all sessions kept).

    Returns:
        A :class:`SampledSessions` carrying the sampled sessions plus
        the pre-sample total and dropped count.

    Raises:
        OSError: ``csv_path`` cannot be opened.
    """
    grouped: dict[str, list[tuple[int, int, ClickstreamEvent]]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            session_id = (row.get("session_id") or "").strip()
            if not session_id:
                continue
            if store_id is not None and (row.get("store_id") or "").strip() != store_id:
                continue
            event = _event_from_row(session_id, row)
            grouped.setdefault(session_id, []).append(
                (event.timestamp, row_index, event),
            )

    total = len(grouped)
    sampled_ids = sorted(grouped)
    if max_sessions > 0:
        sampled_ids = sampled_ids[:max_sessions]

    sessions = tuple(
        Session(
            session_id=session_id,
            events=tuple(event for _, _, event in sorted(grouped[session_id])),
        )
        for session_id in sampled_ids
    )
    return SampledSessions(
        sessions=sessions,
        total_sessions=total,
        dropped=total - len(sessions),
    )


def _event_from_row(session_id: str, row: dict[str, str | None]) -> ClickstreamEvent:
    """Build a :class:`ClickstreamEvent` from one CSV row."""
    return ClickstreamEvent(
        session_id=session_id,
        timestamp=_parse_timestamp(row.get("timestamp")),
        semantic_action=(row.get("semantic_action") or "").strip().lower(),
        raw_action=(row.get("raw_action") or "").strip(),
        url=(row.get("url") or "").strip(),
        product_handle=(row.get("product_handle") or "").strip(),
        landing_url=(row.get("landing_url") or "").strip(),
    )


def _parse_timestamp(raw: str | None) -> int:
    """Parse ``timestamp`` as integer milliseconds; ``0`` on failure.

    Failures keep their original CSV order via the stable secondary key
    applied by :func:`load_sessions`, so a malformed timestamp never
    reorders an otherwise-ordered session.
    """
    if raw is None:
        return 0
    stripped = raw.strip()
    try:
        return int(stripped)
    except ValueError:
        try:
            # Tolerate float-milliseconds (e.g. "1699999999999.0").
            return int(float(stripped))
        except ValueError:
            return 0


def _extract_path(raw: str) -> str | None:
    """Return a root-relative replay path from ``raw``, or ``None``.

    Accepts a bare path (``/products/foo``) verbatim, or extracts the
    path (plus query) from an absolute ``http(s)`` URL. Anything else
    (empty string, fragment-only, opaque scheme) yields ``None``.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.startswith("/"):
        return stripped
    parsed = urlparse(stripped)
    if parsed.scheme in _HTTP_SCHEMES and parsed.path:
        return f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    return None


__all__ = [
    "ACTION_ADD",
    "ACTION_DETAIL",
    "ACTION_EXPLORE_STAY",
    "ACTION_TERMINATE",
    "DEFAULT_MAX_SESSIONS",
    "ClickstreamEvent",
    "SampledSessions",
    "Session",
    "load_sessions",
]
