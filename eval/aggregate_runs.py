"""Aggregate evaluation results across multiple runs of the *same setting*.

Several runs of one configuration each produce their own ``eval_results.json``.
The headline metrics (``js_divergence``, ``chi_square``, ``p_value``,
transition-matrix distances, embedding similarities, …) are computed from the raw
session *distributions*, so they cannot be averaged across the per-run JSON files.
Instead this module **pools the raw sessions from every run and recomputes the
evaluation once**.

This is possible because ``run_eval.run_evaluation`` persists the fully-normalized
inputs into every ``eval_results.json`` (``real_sessions`` / ``synth_sessions``,
each an ``asdict(NormalizedSession)`` carrying ``product_texts`` etc.). So pooling
needs no product catalog and no generation re-run — only the eval-side rationale
LLM step runs again.

Every evaluator pairs real↔synth by ``session_id`` (via ``_by_id`` dicts), and runs
of the same setting reuse the *same* real session_ids. Pooling with the original
ids would make those dicts silently keep only one run's session per id. So each
run's session_ids are prefixed with a per-run tag to make every (run, session)
globally unique while preserving within-run real↔synth pairing.

As a CLI:
    python -m eval.aggregate_runs \
        --runs results_toy_0625/traj_cond/<slug>/1 \
               results_toy_0625/traj_cond/<slug>/2 \
        --output results_toy_0625/traj_cond/<slug>/eval_results.json

As a library:
    from eval.aggregate_runs import aggregate_runs
    results = aggregate_runs(run_paths=[...], output=..., llm_config=...)
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, fields
from pathlib import Path

from eval.evaluation_aggregator import EvaluationAggregator
from eval.models import NormalizedSession
from eval.run_eval import (
    _load_eval_llm_config,
    _DEFAULT_CONFIG,
    write_simplified_results,
)

logger = logging.getLogger(__name__)

_NORMALIZED_FIELDS = {f.name for f in fields(NormalizedSession)}


def _resolve_eval_results_path(run_path: str) -> Path:
    """Map a run location to its ``eval_results.json``.

    Accepts either a direct path to an ``eval_results.json`` file or a run folder
    that contains one.
    """
    p = Path(run_path)
    if p.is_file():
        return p
    candidate = p / "eval_results.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No eval_results.json found at {run_path}")


def _load_sessions(records: list[dict], tag: str) -> list[NormalizedSession]:
    """Rebuild NormalizedSession objects from stored dicts, prefixing each
    ``session_id`` with ``<tag>__`` so pooled (run, session) pairs stay unique."""
    sessions = []
    for rec in records:
        kwargs = {k: v for k, v in rec.items() if k in _NORMALIZED_FIELDS}
        kwargs["session_id"] = f"{tag}__{kwargs.get('session_id', '')}"
        sessions.append(NormalizedSession(**kwargs))
    return sessions


def _merge_api_cost(blocks: list[tuple[str, dict]]) -> dict:
    """Sum the per-run ``api_cost`` blocks. ``blocks`` is a list of (tag, block)."""
    merged = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_time_elapse_seconds": 0.0,
        "total_price_cost_usd": 0.0,
        "num_api_calls": 0,
        "num_sessions": 0,
        "model_names": set(),
        "per_session": {},
    }
    for tag, block in blocks:
        if not block:
            continue
        merged["total_prompt_tokens"] += block.get("total_prompt_tokens", 0)
        merged["total_completion_tokens"] += block.get("total_completion_tokens", 0)
        merged["total_time_elapse_seconds"] += block.get("total_time_elapse_seconds", 0.0)
        merged["total_price_cost_usd"] += block.get("total_price_cost_usd", 0.0)
        merged["num_api_calls"] += block.get("num_api_calls", 0)
        merged["num_sessions"] += block.get("num_sessions", 0)
        merged["model_names"].update(block.get("model_names", []) or [])
        for sid, stats in (block.get("per_session", {}) or {}).items():
            merged["per_session"][f"{tag}__{sid}"] = stats
    merged["model_names"] = sorted(merged["model_names"])
    merged["total_time_elapse_seconds"] = round(merged["total_time_elapse_seconds"], 2)
    merged["total_price_cost_usd"] = round(merged["total_price_cost_usd"], 6)
    return merged


def _merge_post_verify(blocks: list[tuple[str, dict]]) -> dict:
    """Combine per-run ``post_verify`` blocks (sum exclusions, keep thresholds)."""
    thresholds = sorted({b.get("threshold") for _, b in blocks if b and b.get("threshold") is not None})
    excluded = []
    for tag, block in blocks:
        for rec in (block or {}).get("excluded_sessions", []) or []:
            rec = dict(rec)
            if "session_id" in rec:
                rec["session_id"] = f"{tag}__{rec['session_id']}"
            excluded.append(rec)
    return {
        "threshold": thresholds[0] if len(thresholds) == 1 else thresholds,
        "num_excluded": len(excluded),
        "excluded_sessions": excluded,
    }


def aggregate_runs(
    run_paths: list[str],
    output: str,
    llm_config=None,
    verbose: bool = False,
) -> dict:
    """Pool sessions from every run in ``run_paths`` and recompute the evaluation.

    Each entry in ``run_paths`` is either an ``eval_results.json`` file or a folder
    containing one. Writes the combined result to ``output`` and returns it.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    if not run_paths:
        raise ValueError("aggregate_runs requires at least one run path")

    pooled_real: list[NormalizedSession] = []
    pooled_synth: list[NormalizedSession] = []
    api_cost_blocks: list[tuple[str, dict]] = []
    post_verify_blocks: list[tuple[str, dict]] = []
    sources: list[dict] = []
    single_run_data: dict | None = None

    for idx, run_path in enumerate(run_paths, start=1):
        results_path = _resolve_eval_results_path(run_path)
        with open(results_path) as f:
            data = json.load(f)
        if len(run_paths) == 1:
            single_run_data = data

        # Tag from the run folder name when meaningful (e.g. "1", "2"), else index.
        folder_name = results_path.parent.name
        tag = f"run{idx}_{folder_name}"

        real = _load_sessions(data.get("real_sessions", []), tag)
        synth = _load_sessions(data.get("synth_sessions", []), tag)
        pooled_real.extend(real)
        pooled_synth.extend(synth)

        api_cost_blocks.append((tag, data.get("api_cost", {})))
        post_verify_blocks.append((tag, data.get("post_verify", {})))
        sources.append(
            {
                "path": str(results_path),
                "tag": tag,
                "n_real_sessions": len(real),
                "n_synth_sessions": len(synth),
            }
        )
        logger.info(
            "Loaded %s: %d real, %d synth sessions",
            results_path,
            len(real),
            len(synth),
        )

    if not pooled_real or not pooled_synth:
        raise ValueError(
            "Cannot aggregate: pooled data has zero real or synthetic sessions"
        )

    logger.info(
        "Pooled %d real and %d synthetic sessions across %d run(s)",
        len(pooled_real),
        len(pooled_synth),
        len(run_paths),
    )

    if single_run_data is not None:
        # A single run's metrics are unaffected by session-id tagging, so reuse
        # them as-is and only rewrite the pooled/tagged blocks below.
        logger.info("Single run: reusing its metrics without recomputation")
        results = single_run_data
    else:
        aggregator = EvaluationAggregator(llm_config=llm_config)
        results = aggregator.run(pooled_real, pooled_synth)

    results["api_cost"] = _merge_api_cost(api_cost_blocks)
    results["post_verify"] = _merge_post_verify(post_verify_blocks)
    results.setdefault("metadata", {})
    results["metadata"]["num_runs_aggregated"] = len(run_paths)
    results["metadata"]["sources"] = sources

    results["real_sessions"] = [asdict(s) for s in pooled_real]
    results["synth_sessions"] = [asdict(s) for s in pooled_synth]

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Aggregated results written to %s", output)
    write_simplified_results(results, output)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pool sessions from multiple same-setting runs and recompute "
        "the evaluation once."
    )
    parser.add_argument(
        "--runs",
        required=True,
        nargs="+",
        help="Run folders (each containing eval_results.json) or direct paths to "
        "eval_results.json files.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="YAML config to read the eval LLM settings from (default %(default)s).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the LLM base_url used for rationale generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM model used for rationale generation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    llm_config = _load_eval_llm_config(
        config_path=args.config, base_url=args.base_url, model=args.model
    )
    aggregate_runs(
        run_paths=args.runs,
        output=args.output,
        llm_config=llm_config,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
