"""CLI + callable entrypoint for the synthetic vs real data evaluation pipeline.

As a CLI:
    python -m eval.run_eval \
        --real-data data/sessions_raw.csv \
        --synthetic-data results/action_follow/session_data_2026-04-21_13-22-00.json \
        --product-catalog data/products.csv \
        --output results/action_follow/eval_results.json

    `--product-catalog` is used for both sides; 
    pass `--real-product-catalog` / `--synthetic-product-catalog` to override a single side.

As a library:
    from eval.run_eval import run_evaluation
    results = run_evaluation(real_data=..., synthetic_data=..., output=...)
"""

import argparse
import json
import logging
import os
import sys
import types
from dataclasses import asdict
from pathlib import Path

import yaml

from eval.parsers import (
    RealSessionParser,
    SyntheticSessionParser,
    load_api_cost_summary,
)
from eval.evaluation_aggregator import EvaluationAggregator

_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "output", "eval_results.json")
_DEFAULT_POST_VERIFY_THRESHOLD = 0.8
_DEFAULT_CONFIG = "conf/base.yaml"

logger = logging.getLogger(__name__)


def _simplified_output_path(output: str) -> str:
    """Derive the simplified-report path from the raw output path
    (``eval_results.json`` -> ``eval_results_simplified.json``)."""
    p = Path(output)
    return str(p.with_name(p.stem + "_simplified" + p.suffix))


def build_simplified_results(results: dict) -> dict:
    """Flatten the nested evaluation report into a compact set of headline
    metrics. Missing/skipped blocks yield ``None`` rather than raising."""
    seq = results.get("sequence_level", {}) or {}
    out = results.get("outcome_level", {}) or {}
    sem = results.get("semantic_level", {}) or {}
    return {
        "outcome_jsd": out.get("js_divergence"),
        "action_freq_jsd": (seq.get("action_frequency") or {}).get("js_divergence"),
        "session_length_jsd": (seq.get("session_length") or {}).get("js_divergence"),
        "transition_matrix_L1": (seq.get("transition_matrix") or {}).get("l1_distance"),
        "trajectory_lev_distance": (seq.get("trajectory_difference") or {}).get(
            "edit_distance_avg"
        ),
        "product_coherence_gap": sem.get("product_coherence_gap"),
        "product_diversity_ratio": sem.get("product_diversity_ratio"),
        "product_sim_per_session": (
            sem.get("cross_product_similarity_per_session") or {}
        ).get("mean_nn_similarity"),
        "rationale_similarity": (sem.get("rationale_comparison") or {}).get(
            "mean_paired_similarity"
        ),
    }


def write_simplified_results(results: dict, output: str) -> str:
    """Write the simplified report next to the raw ``output`` path and return it."""
    simplified_path = _simplified_output_path(output)
    with open(simplified_path, "w") as f:
        json.dump(build_simplified_results(results), f, indent=2)
    logger.info("Simplified results written to %s", simplified_path)
    return simplified_path


def _load_eval_llm_config(config_path=_DEFAULT_CONFIG, base_url=None, model=None):
    """Build the LLM config (base_url + model) used by the rationale evaluation.

    Reads `eval.llm` from the YAML config, falling back to the top-level `llm`
    block, then applies any explicit CLI overrides. Only the `llm`/`eval`
    sections are read — they carry no OmegaConf interpolations, so plain
    `yaml.safe_load` is safe. Returns a SimpleNamespace with `.base_url`/`.model`
    (what SemanticLevelFidelity reads via getattr)."""
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    llm = dict((cfg.get("eval", {}) or {}).get("llm") or cfg.get("llm", {}) or {})
    if base_url is not None:
        llm["base_url"] = base_url
    if model is not None:
        llm["model"] = model
    return types.SimpleNamespace(**llm)


def _compute_post_verify_exclusions(
    synthetic_data: str, threshold: float
) -> tuple[set[str], list[dict]]:
    """Scan ``<gen_dir>/runs/*`` for ``post_verify_result.json`` files and decide
    which sessions to drop from evaluation.

    Each run dir holds a ``basic_info.json`` (carrying the ``session_id``) and,
    for verifier runs, a ``post_verify_result.json`` whose dict-valued entries
    (e.g. ``trajectory_consistency``, ``realism``) each carry a numeric
    ``score``. A run's aggregate score is the mean of those scores; if it falls
    below ``threshold`` the session is excluded.

    A run with no ``post_verify_result.json`` is kept (skipped silently), so this
    is a no-op for non-verifier experiments.

    Returns ``(excluded_ids, excluded_records)`` where each record is
    ``{session_id, avg_score, scores}``.
    """
    runs_dir = Path(synthetic_data).parent / "runs"
    if not runs_dir.is_dir():
        return set(), []

    excluded_ids: set[str] = set()
    excluded_records: list[dict] = []

    for run_dir in sorted(runs_dir.iterdir()):
        verify_path = run_dir / "post_verify_result.json"
        if not verify_path.is_file():
            continue  # not a verifier run (or not yet verified) — keep it
        try:
            with open(run_dir / "basic_info.json") as f:
                session_id = str(json.load(f).get("session_id", "")).strip()
            with open(verify_path) as f:
                verify = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping post-verify for %s: %s", run_dir.name, e)
            continue

        if not session_id:
            logger.warning("No session_id in %s/basic_info.json; keeping", run_dir.name)
            continue

        scores = {
            k: v["score"]
            for k, v in verify.items()
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float))
        }
        if not scores:
            continue  # nothing scorable — keep

        avg_score = sum(scores.values()) / len(scores)
        if avg_score < threshold:
            excluded_ids.add(session_id)
            excluded_records.append(
                {"session_id": session_id, "avg_score": avg_score, "scores": scores}
            )

    return excluded_ids, excluded_records


def run_evaluation(
    real_data: str,
    synthetic_data: str,
    synthetic_product_catalog: str | None = None,
    real_product_catalog: str | None = None,
    product_catalog: str | None = None,
    output: str = _DEFAULT_OUTPUT,
    verbose: bool = False,
    llm_config=None,
    post_verify_threshold: float = _DEFAULT_POST_VERIFY_THRESHOLD,
) -> dict:
    """Run the full evaluation and write results to `output`.

    `product_catalog` is a fallback used for whichever side does not have a
    dedicated `real_product_catalog` / `synthetic_product_catalog`.

    Returns the results dict (already augmented with api_cost).
    Raises SystemExit(1) if either side has zero usable sessions.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    real_catalog = real_product_catalog or product_catalog
    synth_catalog = synthetic_product_catalog or product_catalog

    real_sessions = RealSessionParser(
        product_catalog=real_catalog
    ).parse(real_data)
    synth_sessions = SyntheticSessionParser(
        product_catalog=synth_catalog
    ).parse(synthetic_data)
    logger.info(
        "Parsed %d real, %d synthetic sessions",
        len(real_sessions),
        len(synth_sessions),
    )

    excluded_ids, excluded_records = _compute_post_verify_exclusions(
        synthetic_data, post_verify_threshold
    )
    if excluded_ids:
        synth_sessions = [s for s in synth_sessions if s.session_id not in excluded_ids]
        logger.info(
            "Excluded %d session(s) below post-verify threshold %.2f",
            len(excluded_ids),
            post_verify_threshold,
        )

    synth_ids = {s.session_id for s in synth_sessions}
    real_sessions = [s for s in real_sessions if s.session_id in synth_ids]
    logger.info(
        "Filtered real sessions to %d matching synthetic session IDs",
        len(real_sessions),
    )

    if not real_sessions or not synth_sessions:
        logger.error("Cannot evaluate: need at least 1 session in each dataset")
        sys.exit(1)

    aggregator = EvaluationAggregator(llm_config=llm_config)
    results = aggregator.run(real_sessions, synth_sessions)

    cost = load_api_cost_summary(synthetic_data)
    results["api_cost"] = {
        "total_prompt_tokens": cost.total_prompt_tokens,
        "total_completion_tokens": cost.total_completion_tokens,
        "total_time_elapse_seconds": round(cost.total_time_elapse, 2),
        "total_price_cost_usd": round(cost.total_price_cost, 6),
        "num_api_calls": cost.num_api_calls,
        "num_sessions": cost.num_sessions,
        "model_names": sorted(cost.model_names),
        "per_session": cost.per_session,
    }
    print("\n--- API Cost Summary ---")
    print(f"  Sessions:           {cost.num_sessions}")
    print(f"  Total API calls:    {cost.num_api_calls}")
    print(f"  Models:             {', '.join(sorted(cost.model_names)) or 'n/a'}")
    print(f"  Prompt tokens:      {cost.total_prompt_tokens}")
    print(f"  Completion tokens:  {cost.total_completion_tokens}")
    print(f"  Total time:         {round(cost.total_time_elapse, 2)}s")
    print(f"  Total cost:         ${cost.total_price_cost:.6f}")

    results["post_verify"] = {
        "threshold": post_verify_threshold,
        "num_excluded": len(excluded_ids),
        "excluded_sessions": excluded_records,
    }

    results["real_sessions"] = [asdict(s) for s in real_sessions]
    results["synth_sessions"] = [asdict(s) for s in synth_sessions]

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", output)
    write_simplified_results(results, output)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic session data against real clickstream data"
    )
    parser.add_argument("--real-data", required=True, help="Path to real clickstream CSV")
    parser.add_argument(
        "--synthetic-data", required=True, help="Path to synthetic session JSON"
    )
    parser.add_argument(
        "--synthetic-product-catalog",
        default=None,
        help="Path to product catalog CSV used to enrich synthetic sessions",
    )
    parser.add_argument(
        "--real-product-catalog",
        default=None,
        help="Path to product catalog CSV (keyed by product_handle) used to "
        "build product_texts for real sessions",
    )
    parser.add_argument(
        "--product-catalog",
        default=None,
        help="Path to a product catalog CSV used for both real and synthetic "
        "sessions when the per-side flags are not provided",
    )
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument(
        "--post-verify-threshold",
        type=float,
        default=_DEFAULT_POST_VERIFY_THRESHOLD,
        help="Exclude runs whose mean post_verify_result.json score is below this "
        "threshold (default %(default)s). Runs without the file are kept.",
    )
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
        default="gpt-5.4",
        help="Override the LLM model used for rationale generation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    llm_config = _load_eval_llm_config(
        config_path=args.config, base_url=args.base_url, model=args.model
    )

    run_evaluation(
        real_data=args.real_data,
        synthetic_data=args.synthetic_data,
        synthetic_product_catalog=args.synthetic_product_catalog,
        real_product_catalog=args.real_product_catalog,
        product_catalog=args.product_catalog,
        output=args.output,
        verbose=args.verbose,
        llm_config=llm_config,
        post_verify_threshold=args.post_verify_threshold,
    )


if __name__ == "__main__":
    main()
