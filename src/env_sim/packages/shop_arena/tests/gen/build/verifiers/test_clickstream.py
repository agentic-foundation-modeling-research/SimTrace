"""Unit tests for the pure clickstream CSV layer (``_clickstream``).

Covers parsing, session grouping, timestamp ordering (with stable ties),
optional ``store_id`` filtering, deterministic session sampling / cap, and
the path → twin-URL mapping. No browser or dev server is involved — this
module is the value layer behind
:class:`~shop_arena.gen.build.verifiers.clickstream_replay.ClickstreamReplayVerifier`.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from shop_arena.gen.build.verifiers._clickstream import (
    ClickstreamEvent,
    load_sessions,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_HEADER: tuple[str, ...] = (
    "session_id",
    "timestamp",
    "semantic_action",
    "raw_action",
    "url",
    "product_handle",
    "store_id",
    "landing_url",
)


def _row(
    *,
    session_id: str,
    timestamp: str = "1",
    semantic_action: str = "detail",
    raw_action: str = "",
    url: str = "",
    product_handle: str = "",
    store_id: str = "",
    landing_url: str = "",
) -> dict[str, str]:
    """Build one CSV row mapping (all columns present)."""
    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "semantic_action": semantic_action,
        "raw_action": raw_action,
        "url": url,
        "product_handle": product_handle,
        "store_id": store_id,
        "landing_url": landing_url,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> Path:
    """Write ``rows`` as a clickstream CSV at ``path`` and return it."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_HEADER))
        writer.writeheader()
        writer.writerows(rows)
    return path


# --------------------------------------------------------------------------- #
# Grouping + ordering
# --------------------------------------------------------------------------- #


def test_load_sessions_groups_rows_by_session_id(tmp_path: Path) -> None:
    """Rows sharing a ``session_id`` collapse into one session."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id="s1", timestamp="1"),
            _row(session_id="s1", timestamp="2"),
            _row(session_id="s2", timestamp="1"),
        ],
    )

    sampled = load_sessions(csv_path)

    assert sampled.total_sessions == 2
    assert [s.session_id for s in sampled.sessions] == ["s1", "s2"]
    assert len(sampled.sessions[0].events) == 2


def test_load_sessions_orders_events_by_timestamp(tmp_path: Path) -> None:
    """Events within a session are ordered by ascending integer timestamp."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id="s1", timestamp="30", raw_action="c"),
            _row(session_id="s1", timestamp="10", raw_action="a"),
            _row(session_id="s1", timestamp="20", raw_action="b"),
        ],
    )

    (session,) = load_sessions(csv_path).sessions

    assert [e.timestamp for e in session.events] == [10, 20, 30]
    assert [e.raw_action for e in session.events] == ["a", "b", "c"]


def test_load_sessions_breaks_timestamp_ties_by_csv_order(tmp_path: Path) -> None:
    """Equal timestamps preserve original CSV row order (stable sort)."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id="s1", timestamp="5", raw_action="first"),
            _row(session_id="s1", timestamp="5", raw_action="second"),
            _row(session_id="s1", timestamp="5", raw_action="third"),
        ],
    )

    (session,) = load_sessions(csv_path).sessions

    assert [e.raw_action for e in session.events] == ["first", "second", "third"]


def test_load_sessions_unparsable_timestamp_sorts_as_zero(tmp_path: Path) -> None:
    """A malformed timestamp sorts as ``0`` without reordering its neighbours."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id="s1", timestamp="7", raw_action="later"),
            _row(session_id="s1", timestamp="oops", raw_action="zero"),
        ],
    )

    (session,) = load_sessions(csv_path).sessions

    assert [e.timestamp for e in session.events] == [0, 7]
    assert [e.raw_action for e in session.events] == ["zero", "later"]


def test_load_sessions_accepts_float_millisecond_timestamp(tmp_path: Path) -> None:
    """A float-milliseconds timestamp is truncated to int, not dropped to 0."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [_row(session_id="s1", timestamp="1699999999999.0")],
    )

    (session,) = load_sessions(csv_path).sessions

    assert session.events[0].timestamp == 1699999999999


def test_load_sessions_skips_rows_without_session_id(tmp_path: Path) -> None:
    """Rows with a blank ``session_id`` are dropped entirely."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id=""),
            _row(session_id="   "),
            _row(session_id="s1"),
        ],
    )

    sampled = load_sessions(csv_path)

    assert sampled.total_sessions == 1
    assert sampled.sessions[0].session_id == "s1"


# --------------------------------------------------------------------------- #
# store_id filter
# --------------------------------------------------------------------------- #


def test_load_sessions_filters_by_store_id(tmp_path: Path) -> None:
    """Only rows whose ``store_id`` matches the filter are retained."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id="s1", store_id="shop-a"),
            _row(session_id="s2", store_id="shop-b"),
        ],
    )

    sampled = load_sessions(csv_path, store_id="shop-a")

    assert [s.session_id for s in sampled.sessions] == ["s1"]


def test_load_sessions_no_store_filter_keeps_all(tmp_path: Path) -> None:
    """Without a ``store_id`` filter every store's rows are kept."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [
            _row(session_id="s1", store_id="shop-a"),
            _row(session_id="s2", store_id="shop-b"),
        ],
    )

    assert load_sessions(csv_path).total_sessions == 2


# --------------------------------------------------------------------------- #
# Deterministic sampling / cap
# --------------------------------------------------------------------------- #


def test_load_sessions_caps_to_first_n_by_sorted_id(tmp_path: Path) -> None:
    """The cap keeps the first N sessions by *sorted* id, deterministically."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [_row(session_id=f"s{i}") for i in (3, 1, 4, 2, 5)],
    )

    sampled = load_sessions(csv_path, max_sessions=2)

    assert [s.session_id for s in sampled.sessions] == ["s1", "s2"]
    assert sampled.total_sessions == 5
    assert sampled.dropped == 3


def test_load_sessions_non_positive_cap_disables_limit(tmp_path: Path) -> None:
    """A non-positive cap keeps every session (no drop)."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [_row(session_id=f"s{i}") for i in range(4)],
    )

    sampled = load_sessions(csv_path, max_sessions=0)

    assert len(sampled.sessions) == 4
    assert sampled.dropped == 0


def test_load_sessions_dropped_zero_when_under_cap(tmp_path: Path) -> None:
    """Fewer sessions than the cap means nothing is dropped."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [_row(session_id="s1"), _row(session_id="s2")],
    )

    assert load_sessions(csv_path, max_sessions=15).dropped == 0


# --------------------------------------------------------------------------- #
# Field normalisation
# --------------------------------------------------------------------------- #


def test_load_sessions_normalises_semantic_action(tmp_path: Path) -> None:
    """``semantic_action`` is lower-cased and stripped."""
    csv_path = _write_csv(
        tmp_path / "clickstream.csv",
        [_row(session_id="s1", semantic_action="  ADD  ")],
    )

    (session,) = load_sessions(csv_path).sessions

    assert session.events[0].semantic_action == "add"


# --------------------------------------------------------------------------- #
# URL mapping (ClickstreamEvent)
# --------------------------------------------------------------------------- #


def _event(*, url: str = "", landing_url: str = "") -> ClickstreamEvent:
    """Build a bare event for URL-mapping assertions."""
    return ClickstreamEvent(
        session_id="s1",
        timestamp=1,
        semantic_action="detail",
        raw_action="",
        url=url,
        product_handle="",
        landing_url=landing_url,
    )


def test_target_path_prefers_url_column() -> None:
    """``target_path`` returns the ``url`` column verbatim when present."""
    event = _event(url="/products/blue-shirt", landing_url="https://live/x")
    assert event.target_path() == "/products/blue-shirt"


def test_target_path_falls_back_to_landing_url_path() -> None:
    """With no ``url``, the path component of ``landing_url`` is used."""
    event = _event(landing_url="https://live.example.com/collections/tops?page=2")
    assert event.target_path() == "/collections/tops?page=2"


def test_target_path_none_when_no_path_available() -> None:
    """An event carrying neither a path nor an http landing URL yields ``None``."""
    assert _event().target_path() is None
    assert _event(landing_url="mailto:x@y.z").target_path() is None


def test_target_url_joins_onto_base_url() -> None:
    """``target_url`` joins the path onto the base URL, collapsing the slash."""
    event = _event(url="/products/blue-shirt")
    assert event.target_url("http://127.0.0.1:3000/") == "http://127.0.0.1:3000/products/blue-shirt"


def test_target_url_none_when_no_path() -> None:
    """``target_url`` is ``None`` when the event has no path (e.g. terminate)."""
    assert _event().target_url("http://127.0.0.1:3000") is None
