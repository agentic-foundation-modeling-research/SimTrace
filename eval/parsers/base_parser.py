from abc import ABC, abstractmethod
from collections import defaultdict

from eval.models import NormalizedSession


class BaseParser(ABC):
    """Abstract base class for session data parsers."""

    @abstractmethod
    def parse(self, filepath: str) -> list[NormalizedSession]:
        """Parse a data file into a list of NormalizedSession objects."""
        ...

    @staticmethod
    def _group_by_session(
        records: list[dict], key: str = "session_id"
    ) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            sid = record.get(key, "")
            if sid:
                groups[sid].append(record)
        return dict(groups)

    @staticmethod
    def _sort_by_timestamp(records: list[dict], key: str = "timestamp") -> list[dict]:
        return sorted(records, key=lambda r: r.get(key, ""))

    @staticmethod
    def _validate_session(session: NormalizedSession) -> bool:
        """Return True if session has at least one action and consistent field lengths."""
        if session.length == 0:
            return False
        n = session.length
        if len(session.product_categories) != n:
            return False
        if len(session.product_texts) != n:
            return False
        if len(session.timestamps) != n:
            return False
        if len(session.urls) != n:
            return False
        if session.search_queries and len(session.search_queries) != n:
            return False
        return True
