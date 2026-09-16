"""HTML helpers for extracting valid action targets from recorded UI context."""

from __future__ import annotations

import html
import re

_SEMANTIC_ID = re.compile(
    r"(?:parser|data)-semantic-id\s*=\s*(['\"])(.*?)\1",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_semantic_ids(ui: str) -> set[str]:
    """Return parser/data semantic IDs present in a simplified HTML snapshot."""
    return {
        html.unescape(match.group(2)).strip()
        for match in _SEMANTIC_ID.finditer(str(ui or ""))
        if match.group(2).strip()
    }

