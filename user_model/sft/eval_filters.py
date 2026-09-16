"""Lightweight SFT record filters shared by inference and unit tests."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def filter_w0_records(records: list[dict]) -> list[dict]:
    """Keep only the first sliding window from a windowed eval dataset.

    ``session_to_records`` appends ``#w0``, ``#w1``, ... to windowed session
    IDs. Evaluating every overlapping window both duplicates turns and creates
    artificial mid-session cold starts. If the input is not windowed, leave it
    unchanged so inference continues to support one-record-per-session data.
    """
    has_window_suffix = any(
        isinstance(record.get("session_id"), str)
        and "#w" in record["session_id"]
        and record["session_id"].rsplit("#w", 1)[-1].isdigit()
        for record in records
    )
    if not has_window_suffix:
        logger.info("Eval data has no #wN session suffixes; keeping all %d records", len(records))
        return records

    selected = [
        record
        for record in records
        if isinstance(record.get("session_id"), str)
        and record["session_id"].endswith("#w0")
    ]
    if not selected:
        raise ValueError("Eval data is windowed but contains no #w0 records.")
    logger.info(
        "Filtered eval records to #w0 only: %d/%d records retained",
        len(selected),
        len(records),
    )
    return selected
