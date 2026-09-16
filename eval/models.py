from dataclasses import dataclass, field
from typing import Any


SEMANTIC_ACTIONS = [
    "detail",
    "explore-search",
    "explore-goto",
    "explore-back",
    "explore-stay",
    "add",
    "checkout",
    "remove",
    "terminate",
]


@dataclass
class NormalizedSession:
    session_id: str
    actions: list[str] = field(default_factory=list)
    product_categories: list[str] = field(default_factory=list)
    product_texts: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.actions)

    def has_action(self, action: str) -> bool:
        return action in self.actions


@dataclass
class ApiCostSummary:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_time_elapse: float = 0.0
    total_price_cost: float = 0.0
    num_api_calls: int = 0
    num_sessions: int = 0
    model_names: set[str] = field(default_factory=set)
    per_session: dict[str, dict[str, Any]] = field(default_factory=dict)
