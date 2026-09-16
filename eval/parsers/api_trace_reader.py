import json
import logging
import os
from collections import defaultdict

from eval.models import ApiCostSummary

logger = logging.getLogger(__name__)


def load_api_cost_summary(filepath: str) -> ApiCostSummary:
    """Read api_trace files referenced in a synthetic data JSON and aggregate cost metrics.

    Each event in the synthetic data has a ``llm_call`` field listing absolute
    paths to ``api_trace_*.json`` files.  Each trace file contains
    ``prompt_tokens``, ``completion_tokens``, ``time``, and ``price_cost``.
    """
    with open(filepath) as f:
        records = json.load(f)

    # `llm_call` paths are stored relative to the directory containing the
    # synthetic data file (the result dir), not the eval process CWD.
    base_dir = os.path.dirname(os.path.abspath(filepath))

    by_session: dict[str, list[str]] = defaultdict(list)
    for event in records:
        sid = event.get("session_id", "")
        for path in event.get("llm_call", []):
            by_session[sid].append(path)

    summary = ApiCostSummary(num_sessions=len(by_session))

    for sid, trace_paths in by_session.items():
        sess: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "time_elapse": 0.0,
            "price_cost": 0.0,
            "num_calls": 0,
            "model_names": set(),
        }
        for path in trace_paths:
            resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
            try:
                with open(resolved) as f:
                    trace = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                logger.warning("Skipping api_trace file %s: %s", resolved, exc)
                continue

            sess["prompt_tokens"] += trace.get("prompt_tokens", 0)
            sess["completion_tokens"] += trace.get("completion_tokens", 0)
            sess["time_elapse"] += trace.get("time", 0.0)
            sess["price_cost"] += trace.get("price_cost", 0.0)
            sess["num_calls"] += 1
            raw_models = trace.get("model_name", [])
            sess["model_names"].update(
                raw_models if isinstance(raw_models, list) else [raw_models]
            )

        sess["model_names"] = sorted(sess["model_names"])
        summary.per_session[sid] = sess
        summary.model_names.update(sess["model_names"])
        summary.total_prompt_tokens += sess["prompt_tokens"]
        summary.total_completion_tokens += sess["completion_tokens"]
        summary.total_time_elapse += sess["time_elapse"]
        summary.total_price_cost += sess["price_cost"]
        summary.num_api_calls += sess["num_calls"]

    return summary
