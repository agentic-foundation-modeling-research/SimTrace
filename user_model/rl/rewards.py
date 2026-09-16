"""Rewards for turn-level GDPO and constraint-aware GRPO.

Both formulas use ``R_format``, ``R_action``, and one scalar ``R_target``. The
module retains the earlier rationale/UI composite classes for compatibility,
but the turn trainer uses :func:`build_formula_rewards`.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

from user_model.rl.utils import extract_semantic_ids

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {
    "click", "type", "hover", "move", "select", "clear", "key_press",
    "scroll", "goto_url", "back", "forward", "refresh", "new_tab",
    "switch_tab", "close_tab", "terminate",
}
TARGET_ACTIONS = {"click", "type"}
_TOKEN = re.compile(r"\w+", re.UNICODE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class TurnRewardConfig:
    base_weight: float = 0.05
    reasoning_weight: float = 0.45
    target_weight: float = 0.50
    ui_grounding_weight: float = 0.70
    reasoning_similarity_weight: float = 0.30

    ui_judge_enabled: bool = True
    ui_judge_provider: str = "gemini"
    ui_judge_model: str = "gemini-3.5-flash"
    ui_judge_base_url: Optional[str] = None
    ui_judge_temperature: float = 0.0
    ui_judge_timeout: float = 90.0
    ui_judge_max_workers: int = 8
    ui_judge_failure_mode: str = "raise"  # raise | zero
    ui_judge_max_relevant_elements: int = 12

    reasoning_similarity_backend: str = "embedding"  # embedding | token_f1
    reasoning_embedding_provider: str = "gemini"
    reasoning_embedding_model: str = "gemini-embedding-001"
    reasoning_embedding_base_url: Optional[str] = None
    reasoning_embedding_timeout: float = 90.0
    reasoning_similarity_failure_mode: str = "raise"  # raise | token_f1

    def validate(self) -> None:
        for name in (
            "base_weight", "reasoning_weight", "target_weight",
            "ui_grounding_weight", "reasoning_similarity_weight",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        outer = self.base_weight + self.reasoning_weight + self.target_weight
        inner = self.ui_grounding_weight + self.reasoning_similarity_weight
        if outer <= 0 or inner <= 0:
            raise ValueError("Reward weight sums must be positive")
        if self.base_weight + self.reasoning_weight <= 0:
            raise ValueError(
                "base_weight + reasoning_weight must be positive for targetless actions"
            )
        if self.ui_judge_failure_mode not in {"raise", "zero"}:
            raise ValueError("ui_judge_failure_mode must be 'raise' or 'zero'")
        if self.reasoning_similarity_failure_mode not in {"raise", "token_f1"}:
            raise ValueError(
                "reasoning_similarity_failure_mode must be 'raise' or 'token_f1'"
            )


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def parse_completion(text: str) -> Optional[dict[str, Any]]:
    """Parse a completion only when the entire response is one JSON object."""
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return value if isinstance(value, dict) else None


def valid_schema(value: Optional[dict[str, Any]]) -> bool:
    """Validate the operational fields required by the next-action schema."""
    if not value or not isinstance(value.get("rationale"), str):
        return False
    action = value.get("action")
    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        return False
    if action in TARGET_ACTIONS and not isinstance(value.get("target"), str):
        return False
    if action == "type":
        return isinstance(value.get("text"), str) and (
            "enter" not in value or isinstance(value["enter"], bool)
        )
    if action == "select":
        return isinstance(value.get("value"), str)
    if action == "key_press":
        return isinstance(value.get("key"), str) and (
            "target" not in value or isinstance(value["target"], str)
        )
    if action == "scroll":
        return (
            value.get("direction") in {"up", "down"}
            and isinstance(value.get("amount"), (int, float))
            and not isinstance(value.get("amount"), bool)
            and ("target" not in value or isinstance(value["target"], str))
        )
    if action in {"goto_url", "new_tab"}:
        return isinstance(value.get("url"), str)
    if action in {"switch_tab", "close_tab"}:
        return isinstance(value.get("tab_id"), int) and not isinstance(
            value.get("tab_id"), bool
        )
    return True


def _token_f1(left: str, right: str) -> float:
    from collections import Counter

    a = Counter(t.casefold() for t in _TOKEN.findall(left or ""))
    b = Counter(t.casefold() for t in _TOKEN.findall(right or ""))
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    overlap = sum((a & b).values())
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class ReasoningSimilarityScorer:
    def __init__(self, cfg: TurnRewardConfig):
        self.cfg = cfg

    def score_many(self, predicted: list[str], reference: list[str]) -> list[float]:
        if self.cfg.reasoning_similarity_backend == "token_f1":
            return [_token_f1(a, b) for a, b in zip(predicted, reference)]
        if self.cfg.reasoning_similarity_backend != "embedding":
            raise ValueError(
                "reasoning_similarity_backend must be 'embedding' or 'token_f1'"
            )
        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": (
                    f"{self.cfg.reasoning_embedding_provider}/"
                    f"{self.cfg.reasoning_embedding_model}"
                ),
                "input": predicted + reference,
                "timeout": self.cfg.reasoning_embedding_timeout,
            }
            if self.cfg.reasoning_embedding_base_url:
                kwargs["base_url"] = self.cfg.reasoning_embedding_base_url
            response = litellm.embedding(**kwargs)
            vectors = [item["embedding"] for item in response.data]
            n = len(predicted)
            return [
                _cosine_01(vectors[i], vectors[n + i])
                for i in range(n)
            ]
        except Exception:
            if self.cfg.reasoning_similarity_failure_mode == "token_f1":
                logger.exception("Embedding similarity failed; using token F1 fallback")
                return [_token_f1(a, b) for a, b in zip(predicted, reference)]
            raise


def _cosine_01(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if not ln or not rn:
        return 0.0
    # Embedding cosine is already a useful [approximately 0, 1] semantic score.
    return max(0.0, min(1.0, dot / (ln * rn)))


_UI_JUDGE_SYSTEM = """You are a strict factual-grounding judge for a web-action model. Please evaluate whether the rationale is factually grounded in the current UI. CURRENT_UI is the raw simplified HTML from the grounding webpage. Interactive elements may carry parser-semantic-id attributes, and the target is the semantic ID of the element that the model intends to act upon. The rationale is a natural-language explanation of the model's selected action and target.

Please evaluate whether the rationale's claims about the CURRENT UI supported by the simplified HTML? Return only JSON: {"explanation": "<brief reason>", "score": <number from 0 to 1>}.

Examples:
If the current UI contains both search bar and search_icon, and the rationale says "The search field is already visible, so I will enter shoes.".
Return:
{"explanation": "The rationale is grounded in the current UI because it correctly identifies that the search field is visible.", "score": 1.0}.

If the current UI contains both search bar and search_icon, and the rationale says "I need to open the search interface."
Return:
{"explanation": "The rationale is not grounded in the current UI because it incorrectly states that the search interface needs to be opened, while the search bar is already visible.", "score": 0.0}.

If the rationale doesn't mention any elements present in the current UI, or if it makes claims that cannot be verified against the simplified HTML.
Return:
{"explanation": "The rationale does not have a checkable connection to the current UI.", "score": 0.25}.
"""


class GeminiUIGroundingJudge:
    def __init__(
        self,
        cfg: TurnRewardConfig,
        call_fn: Optional[Callable[[list[dict[str, str]]], str]] = None,
    ):
        self.cfg = cfg
        self._call_fn = call_fn or self._call
        # Cache both the score and whether it came from the configured failure
        # fallback.  Keeping the status lets the trainer distinguish a genuine
        # judge score of 0 from an API/JSON failure that was converted to 0.
        self._cache: dict[str, tuple[float, bool]] = {}
        self._lock = threading.Lock()

    def _call(self, messages: list[dict[str, str]]) -> str:
        import litellm

        kwargs: dict[str, Any] = {
            "model": f"{self.cfg.ui_judge_provider}/{self.cfg.ui_judge_model}",
            "messages": messages,
            "max_tokens": 256,
            "timeout": self.cfg.ui_judge_timeout,
            "drop_params": True,
            "response_format": {"type": "json_object"},
            "cache": {"no-cache": True, "no-store": True},
        }
        if self.cfg.ui_judge_base_url:
            kwargs["base_url"] = self.cfg.ui_judge_base_url
        response = litellm.completion(**kwargs)
        return response.choices[0].message["content"]

    def score_with_status(
        self, ui: str, rationale: str, action: str, target: str
    ) -> tuple[float, bool]:
        payload = {
            "predicted_rationale": rationale,
            "predicted_action": action,
            "predicted_target": target,
            "current_ui": ui,
        }
        cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        messages = [
            {"role": "system", "content": _UI_JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            raw = self._call_fn(messages)
            match = _JSON_OBJECT.search(raw or "")
            value = json.loads(match.group(0) if match else raw)
            score = max(0.0, min(1.0, float(value["score"])))
            failed = False
        except Exception:
            if self.cfg.ui_judge_failure_mode == "zero":
                logger.exception("UI judge failed; assigning R_ui=0")
                score = 0.0
                failed = True
            else:
                raise
        with self._lock:
            self._cache[cache_key] = (score, failed)
        return score, failed

    def score(self, ui: str, rationale: str, action: str, target: str) -> float:
        score, _ = self.score_with_status(ui, rationale, action, target)
        return score

    def score_many(self, rows: list[tuple[str, str, str, str]]) -> list[float]:
        scores, _ = self.score_many_with_status(rows)
        return scores

    def score_many_with_status(
        self, rows: list[tuple[str, str, str, str]]
    ) -> tuple[list[float], list[bool]]:
        workers = max(1, min(self.cfg.ui_judge_max_workers, len(rows)))
        if workers == 1:
            results = [self.score_with_status(*row) for row in rows]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(
                    pool.map(lambda row: self.score_with_status(*row), rows)
                )
        return (
            [score for score, _ in results],
            [failed for _, failed in results],
        )


def _aligned_rows(kwargs: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Align dataset columns with TRL's flattened completion group.

    Depending on the TRL release, reward kwargs arrive either already repeated
    to ``batch * num_generations`` or once per source prompt.  GRPO orders the
    latter as G consecutive candidates for each prompt.
    """
    rows = []
    for idx in range(n):
        row = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                if len(value) == n:
                    row[key] = value[idx]
                elif value and n % len(value) == 0:
                    row[key] = value[idx // (n // len(value))]
            elif value is not None:
                row[key] = value
        rows.append(row)
    return rows


@dataclass
class TurnFormulaRewardConfig:
    """Weights used by the constraint-aware reward formula."""

    format_weight: float = 1.0
    action_weight: float = 0.5
    target_weight: float = 0.5

    def validate(self) -> None:
        if min(self.format_weight, self.action_weight, self.target_weight) < 0:
            raise ValueError("reward weights must be non-negative")
        if not math.isclose(
            self.action_weight + self.target_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("action_weight + target_weight must equal 1")


def target_substring_score(predicted: str, ground_truth: str) -> float:
    """Return longest-common-substring length over the targets' average length.

    Matching is case-sensitive. Normalizing by the average length makes the
    score symmetric: swapping predicted and ground truth does not change it.
    """

    predicted = str(predicted or "")
    ground_truth = str(ground_truth or "")
    average_length = (len(predicted) + len(ground_truth)) / 2
    if average_length == 0:
        return 1.0
    match = SequenceMatcher(
        None, predicted, ground_truth, autojunk=False
    ).find_longest_match()
    return match.size / average_length


def target_reward_score(predicted: str, ground_truth: str, ui: str) -> float:
    """Return the single scalar ``R_target`` consumed by GDPO."""

    predicted = str(predicted or "")
    ground_truth = str(ground_truth or "")
    lcs = target_substring_score(predicted, ground_truth)
    valid_ui_target = float(
        bool(predicted) and predicted in extract_semantic_ids(str(ui or ""))
    )
    exact_target = float(predicted == ground_truth)
    return 0.6 * lcs + 0.1 * valid_ui_target + 0.3 * exact_target


def score_formula_components(
    completions: list[Any], **kwargs: Any
) -> list[dict[str, float]]:
    """Score the three reward objectives shared by GDPO and constraint-aware."""

    rows = _aligned_rows(kwargs, len(completions))
    parsed = [
        parse_completion(_completion_text(completion))
        for completion in completions
    ]
    components = []
    for value, row in zip(parsed, rows):
        value = value or {}
        predicted_action = str(value.get("action", ""))
        ground_truth_action = str(row.get("gt_action", ""))
        predicted_target = str(value.get("target", "") or "")
        ground_truth_target = str(row.get("gt_target", "") or "")
        ui = str(row.get("ui", "") or "")
        components.append(
            {
                "format": float(valid_schema(value)),
                "action": float(
                    bool(ground_truth_action)
                    and predicted_action == ground_truth_action
                ),
                "target": target_reward_score(
                    predicted_target, ground_truth_target, ui
                ),
            }
        )
    return components


def _log_formula_components(
    completions: list[Any], components: list[dict[str, float]], kwargs: dict[str, Any]
) -> None:
    """Add shared component and label columns to TRL's completion logs once."""

    log_extra = kwargs.get("log_extra")
    if not callable(log_extra):
        return
    rows = _aligned_rows(kwargs, len(completions))
    predicted = [
        parse_completion(_completion_text(value)) or {}
        for value in completions
    ]
    ground_truth_actions = [str(row.get("gt_action", "")) for row in rows]
    ground_truth_targets = [str(row.get("gt_target", "") or "") for row in rows]

    for key in ("session_id", "turn_idx", "reference_rationale"):
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            log_extra(
                "ground_truth_rationale" if key == "reference_rationale" else key,
                values,
            )
    log_extra("predicted_action", [str(row.get("action", "")) for row in predicted])
    log_extra(
        "predicted_target",
        [str(row.get("target", "") or "") for row in predicted],
    )
    log_extra("ground_truth_action", ground_truth_actions)
    log_extra("ground_truth_target", ground_truth_targets)
    log_extra(
        "ground_truth",
        [
            str(row.get("ground_truth_completion") or "")
            or json.dumps(
                {
                    "rationale": str(row.get("reference_rationale", "")),
                    "action": action,
                    **({"target": target} if target else {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row, action, target in zip(
                rows, ground_truth_actions, ground_truth_targets
            )
        ],
    )
    log_extra("format_valid", [bool(row["format"]) for row in components])
    log_extra("action_match", [bool(row["action"]) for row in components])
    log_extra("target_applicable", [bool(value) for value in ground_truth_targets])
    log_extra("target_score", [row["target"] for row in components])
    log_extra(
        "target_match",
        [
            str(value.get("target", "") or "") == ground_truth
            for value, ground_truth in zip(predicted, ground_truth_targets)
        ],
    )


class TurnComponentReward:
    """A single component callable used by TRL's native GDPO aggregation."""

    def __init__(self, component: str, log_components: bool = False):
        if component not in {"format", "action", "target"}:
            raise ValueError(f"Unknown turn reward component: {component}")
        self.component = component
        self.log_components = log_components
        self.__name__ = f"R_{component}"

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        components = score_formula_components(completions, **kwargs)
        if self.log_components:
            _log_formula_components(completions, components, kwargs)
        return [row[self.component] for row in components]


class ConstraintAwareReward:
    """The requested format-penalized, format-gated composite reward."""

    def __init__(self, cfg: TurnFormulaRewardConfig):
        cfg.validate()
        self.cfg = cfg
        self.__name__ = "constraint_aware"

    def score_components(
        self, completions: list[Any], **kwargs: Any
    ) -> list[dict[str, float]]:
        components = score_formula_components(completions, **kwargs)
        for row in components:
            row["total"] = (
                -self.cfg.format_weight * (1.0 - row["format"])
                + row["format"]
                * (
                    self.cfg.action_weight * row["action"]
                    + self.cfg.target_weight * row["target"]
                )
            )
        return components

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        components = self.score_components(completions, **kwargs)
        _log_formula_components(completions, components, kwargs)
        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            log_extra("zero_reward", [row["total"] == 0.0 for row in components])
        return [row["total"] for row in components]


def build_formula_rewards(
    advantage_formula: str, cfg: TurnFormulaRewardConfig
) -> list[Callable[..., list[float]]]:
    """Build the reward layout expected by each advantage formula."""

    if advantage_formula == "gdpo":
        return [
            TurnComponentReward("format", log_components=True),
            TurnComponentReward("action"),
            TurnComponentReward("target"),
        ]
    if advantage_formula == "constraint-aware":
        cfg.validate()
        return [ConstraintAwareReward(cfg)]
    raise ValueError(
        "advantage_formula must be either 'gdpo' or 'constraint-aware'"
    )


class TurnCompositeReward:
    """One TRL-compatible reward callable implementing the gated reward formula."""

    def __init__(
        self,
        cfg: TurnRewardConfig,
        ui_judge: Optional[GeminiUIGroundingJudge] = None,
        reasoning_scorer: Optional[ReasoningSimilarityScorer] = None,
    ):
        cfg.validate()
        self.cfg = cfg
        self.__name__ = "turn_total"
        self.ui_judge = ui_judge or GeminiUIGroundingJudge(cfg)
        self.reasoning_scorer = reasoning_scorer or ReasoningSimilarityScorer(cfg)

    def score_components(
        self, completions: list[Any], **kwargs: Any
    ) -> list[dict[str, float]]:
        n = len(completions)
        rows = _aligned_rows(kwargs, n)
        parsed = [parse_completion(_completion_text(c)) for c in completions]
        formats = [float(valid_schema(value)) for value in parsed]
        action_scores = []
        for value, row in zip(parsed, rows):
            action = str((value or {}).get("action", ""))
            gt_action = str(row.get("gt_action", ""))
            action_scores.append(float(bool(gt_action) and action == gt_action))

        rationales = [
            str(value.get("rationale", "")) if isinstance(value, dict) else ""
            for value in parsed
        ]
        references = [str(row.get("reference_rationale", "")) for row in rows]
        # The two gates make every downstream component irrelevant.  Besides
        # saving cost, skipping these rows prevents malformed candidates from
        # being sent to the external judge.
        eligible = [
            idx for idx, (fmt, action) in enumerate(zip(formats, action_scores))
            if fmt and action
        ]
        similarity = [0.0] * n
        if eligible:
            eligible_similarity = self.reasoning_scorer.score_many(
                [rationales[idx] for idx in eligible],
                [references[idx] for idx in eligible],
            )
            for idx, score in zip(eligible, eligible_similarity):
                similarity[idx] = score

        judge_inputs = []
        for idx in eligible:
            value, row, rationale = parsed[idx] or {}, rows[idx], rationales[idx]
            value = value or {}
            judge_inputs.append(
                (
                    str(row.get("ui", "")),
                    rationale,
                    str(value.get("action", "")),
                    str(value.get("target", "") or ""),
                )
            )
        ui_scores = [0.0] * n
        judge_failures = [False] * n
        if self.cfg.ui_judge_enabled and judge_inputs:
            score_many_with_status = getattr(
                self.ui_judge, "score_many_with_status", None
            )
            if callable(score_many_with_status):
                eligible_ui_scores, eligible_failures = score_many_with_status(
                    judge_inputs
                )
            else:
                # Preserve compatibility with injected/custom judges that only
                # implement score_many. Such judges have no reported failures.
                eligible_ui_scores = self.ui_judge.score_many(judge_inputs)
                eligible_failures = [False] * len(eligible_ui_scores)
            for idx, score, failed in zip(
                eligible, eligible_ui_scores, eligible_failures
            ):
                ui_scores[idx] = score
                judge_failures[idx] = failed

        components = []
        for value, row, fmt, action_score, ui, sim, judge_failed in zip(
            parsed,
            rows,
            formats,
            action_scores,
            ui_scores,
            similarity,
            judge_failures,
        ):
            value = value or {}
            target = str(value.get("target", "") or "")
            gt_target = str(row.get("gt_target", "") or "")
            target_applicable = bool(gt_target)
            target_score = target_substring_score(target, gt_target)

            inner_sum = self.cfg.ui_grounding_weight + self.cfg.reasoning_similarity_weight
            reasoning = (
                self.cfg.ui_grounding_weight * ui
                + self.cfg.reasoning_similarity_weight * sim
            ) / inner_sum
            # Targetless actions omit and renormalize the target term so every action
            # type retains a [0, 1] maximum reward.
            applicable_sum = self.cfg.base_weight + self.cfg.reasoning_weight
            numerator = self.cfg.base_weight + self.cfg.reasoning_weight * reasoning
            if target_applicable:
                applicable_sum += self.cfg.target_weight
                numerator += self.cfg.target_weight * target_score
            total = fmt * action_score * numerator / applicable_sum
            components.append(
                {
                    "format": fmt,
                    "action": action_score,
                    "target": target_score,
                    "ui": float(ui),
                    "reasoning_similarity": float(sim),
                    "reasoning": float(reasoning),
                    "judge_attempted": float(
                        self.cfg.ui_judge_enabled and bool(fmt and action_score)
                    ),
                    "judge_failed": float(judge_failed),
                    "total": float(total),
                }
            )
        return components

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        components = self.score_components(completions, **kwargs)
        results = [row["total"] for row in components]
        # These identifiers are gathered by TRL alongside completions, rewards,
        # and advantages so the trainer can print complete (possibly
        # cross-process) groups after normalization.
        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            rows = _aligned_rows(kwargs, len(completions))
            for key in ("session_id", "turn_idx", "reference_rationale"):
                values = [row.get(key) for row in rows]
                if any(value is not None for value in values):
                    output_key = (
                        "ground_truth_rationale"
                        if key == "reference_rationale"
                        else key
                    )
                    log_extra(output_key, values)

            ground_truth_actions = [str(row.get("gt_action", "")) for row in rows]
            ground_truth_targets = [
                str(row.get("gt_target", "") or "") for row in rows
            ]
            predicted = [parse_completion(_completion_text(c)) or {} for c in completions]
            log_extra("predicted_action", [str(row.get("action", "")) for row in predicted])
            log_extra(
                "predicted_target",
                [str(row.get("target", "") or "") for row in predicted],
            )
            log_extra("ground_truth_action", ground_truth_actions)
            log_extra("ground_truth_target", ground_truth_targets)
            log_extra(
                "ground_truth",
                [
                    str(row.get("ground_truth_completion") or "")
                    or json.dumps(
                        {
                            "rationale": str(row.get("reference_rationale", "")),
                            "action": action,
                            **({"target": target} if target else {}),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for row, action, target in zip(
                        rows, ground_truth_actions, ground_truth_targets
                    )
                ],
            )

            table_columns = {
                "format_valid": [bool(row["format"]) for row in components],
                "action_match": [bool(row["action"]) for row in components],
                # target_applicable makes False unambiguous for actions whose
                # reward intentionally has no target component.
                "target_applicable": [bool(target) for target in ground_truth_targets],
                "target_match": [row["target"] == 1.0 for row in components],
                "ui_judge_reward": [row["ui"] for row in components],
                "reasoning_reward": [row["reasoning"] for row in components],
                "reasoning_similarity": [
                    row["reasoning_similarity"] for row in components
                ],
                "ui_judge_attempted": [
                    bool(row["judge_attempted"]) for row in components
                ],
                "ui_judge_failed": [
                    bool(row["judge_failed"]) for row in components
                ],
                "zero_reward": [row["total"] == 0.0 for row in components],
            }
            for key, values in table_columns.items():
                log_extra(key, values)
        return results
