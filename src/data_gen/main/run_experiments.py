"""Single entrypoint to run a sequence of full-config experiments.

Each experiment in conf/experiments.yaml carries its own `llm`, `simulation_config`, run config and `eval` settings. 
For every experiment this script runs generation and, optionally, evaluation against real data.

Results land in:
    <output_dir>/<agent_mode>/<auto-config-slug>/runs/<timestamp>/
with a config_used.yaml snapshot per folder and one line per run appended to
    <output_dir>/runs_index.jsonl

Usage:
    python -m src.data_gen.main.run_experiments
    python -m src.data_gen.main.run_experiments --config conf/experiments.yaml
    python -m src.data_gen.main.run_experiments --no-run-eval
    python -m src.data_gen.main.run_experiments --continue
"""

import argparse
import logging
import os
import types

from .run import build_config_slug, deep_merge, load_config, run_generation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# agent_mode -> the run-config file whose fields each experiment's `run:` block mirrors.
_ACTION_MODES = ("action_follow", "traj_cond", "persona", "uxagent_traj")


def _assemble_configs(experiments_cfg: dict, exp: dict) -> tuple[dict, dict]:
    """Build the (base_cfg, run_cfg) pair for one experiment by layering, in order:
    conf/base.yaml -> config-wide defaults -> this experiment's overrides."""
    general = load_config("conf/base.yaml")

    sim_defaults = experiments_cfg.get("simulation_config", {}) or {}
    sim_exp = exp.get("simulation_config", {}) or {}
    simulation_config = deep_merge(sim_defaults, sim_exp)
    # The results root is shared across all experiments.
    if experiments_cfg.get("output_dir") is not None:
        simulation_config["output_dir"] = experiments_cfg["output_dir"]

    llm = deep_merge(experiments_cfg.get("llm", {}) or {}, exp.get("llm", {}) or {})

    base_cfg = deep_merge(
        general,
        {"llm": llm, "simulation_config": simulation_config},
    )

    run_cfg = deep_merge(experiments_cfg.get("run", {}) or {}, exp.get("run", {}) or {})
    return base_cfg, run_cfg


def _resolve_eval(experiments_cfg: dict, exp: dict, args) -> dict:
    """Effective eval settings: config-wide `eval:` <- experiment `eval:` <- CLI flags."""
    settings = deep_merge(experiments_cfg.get("eval", {}) or {}, exp.get("eval", {}) or {})
    if not args.run_eval:
        settings["enabled"] = False
    if args.real_data is not None:
        settings["real_data"] = args.real_data
    if args.product_catalog is not None:
        settings["product_catalog"] = args.product_catalog
    if args.post_verify_threshold is not None:
        settings["post_verify_threshold"] = args.post_verify_threshold
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="conf/experiments.yaml",
        help="Path to the experiments config (default: conf/experiments.yaml).",
    )
    parser.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Resume each experiment from where it left off, skipping completed sessions.",
    )
    parser.add_argument(
        "--no-run-eval",
        dest="run_eval",
        action="store_false",
        default=True,
        help="Disable evaluation for all experiments (overrides config).",
    )
    parser.add_argument(
        "--real-data",
        default=None,
        help="Override the eval real-data CSV path for every experiment.",
    )
    parser.add_argument(
        "--product-catalog",
        default=None,
        help="Override the eval product-catalog CSV path for every experiment.",
    )
    parser.add_argument(
        "--post-verify-threshold",
        type=float,
        default=None,
        help="Override the post-verify exclusion threshold for every experiment "
        "(default falls back to the eval config or 0.8).",
    )
    args = parser.parse_args()

    experiments_cfg = load_config(args.config)
    experiments = experiments_cfg.get("experiments") or []
    if not experiments:
        raise ValueError(f"No `experiments` defined in {args.config}")

    # Lazy import so generation-only runs don't pay the eval import cost.
    run_evaluation = None

    seen_dirs: set[str] = set()
    for i, exp in enumerate(experiments):
        name = exp.get("name", f"experiment_{i}")
        base_cfg, run_cfg = _assemble_configs(experiments_cfg, exp)
        agent_mode = base_cfg.get("simulation_config", {}).get("agent_mode")
        slug = build_config_slug(agent_mode, base_cfg, run_cfg)
        top_output_dir = base_cfg.get("simulation_config", {}).get(
            "output_dir", "./results"
        )
        output_dir = os.path.join(top_output_dir, agent_mode, slug)

        print(f"\n{'=' * 60}")
        print(f"Experiment    : {name} ({i + 1}/{len(experiments)})")
        print(f"Agent mode    : {agent_mode}")
        print(f"Provider/model: {base_cfg.get('llm', {}).get('provider')}/"
              f"{base_cfg.get('llm', {}).get('model')}")
        print(f"Output dir    : {output_dir}")
        print("=" * 60)

        if output_dir in seen_dirs:
            log.warning(
                "Skipping '%s': its config resolves to an already-used folder (%s). "
                "Two experiments share the same config knobs — give them distinct settings.",
                name,
                output_dir,
            )
            continue
        seen_dirs.add(output_dir)

        num_runs = int(run_cfg.get("num_runs", 1) or 1)
        eval_cfg = _resolve_eval(experiments_cfg, exp, args)
        eval_enabled = eval_cfg.get("enabled", False)
        real_data = eval_cfg.get("real_data")
        # Prefer a dedicated eval LLM config; fall back to the experiment's own
        # llm (proxy base_url + that experiment's model) when eval.llm is unset.
        eval_llm_config = types.SimpleNamespace(
            **(eval_cfg.get("llm") or base_cfg.get("llm", {}))
        )

        # Repeat the whole setting `num_runs` times; each run nests into <slug>/<i>/.
        per_run_eval_paths: list[str] = []
        for run_index in range(1, num_runs + 1):
            if num_runs > 1:
                print(f"\n--- Run {run_index}/{num_runs} ---")
            try:
                combined_path = run_generation(
                    do_continue=args.do_continue,
                    base_cfg=base_cfg,
                    run_cfg=run_cfg,
                    run_index=run_index,
                )
            except Exception:
                log.exception(
                    "Generation failed for experiment '%s' run %d — continuing.",
                    name, run_index,
                )
                continue

            if combined_path is None:
                log.warning(
                    "Experiment '%s' run %d produced no session records.",
                    name, run_index,
                )
                continue

            if not eval_enabled:
                continue
            if not real_data or not os.path.exists(real_data):
                log.warning(
                    "Skipping eval for '%s': real_data not found (%s).", name, real_data
                )
                continue

            if run_evaluation is None:
                from eval.run_eval import run_evaluation as _re

                run_evaluation = _re

            eval_output = os.path.join(
                os.path.dirname(str(combined_path)), "eval_results.json"
            )
            print(f"Running evaluation → {eval_output}")
            try:
                run_evaluation(
                    real_data=real_data,
                    synthetic_data=str(combined_path),
                    synthetic_product_catalog=eval_cfg.get("synthetic_product_catalog"),
                    real_product_catalog=eval_cfg.get("real_product_catalog"),
                    product_catalog=eval_cfg.get("product_catalog"),
                    output=eval_output,
                    post_verify_threshold=eval_cfg.get("post_verify_threshold", 0.8),
                    llm_config=eval_llm_config,
                )
                per_run_eval_paths.append(eval_output)
            except (Exception, SystemExit):
                log.exception(
                    "Evaluation failed for experiment '%s' run %d — continuing.",
                    name, run_index,
                )

        # Produce the final <slug>/eval_results.json across the runs. aggregate_runs
        # also tags session ids, records metadata.sources, and writes the
        # eval_results_simplified.json sibling; a single run reuses its metrics
        # without recomputation.
        if per_run_eval_paths:
            final_output = os.path.join(output_dir, "eval_results.json")
            from eval.aggregate_runs import aggregate_runs

            print(
                f"Aggregating {len(per_run_eval_paths)} run(s) → {final_output}"
            )
            try:
                aggregate_runs(
                    run_paths=per_run_eval_paths,
                    output=final_output,
                    llm_config=eval_llm_config,
                )
            except (Exception, SystemExit):
                log.exception(
                    "Aggregation failed for experiment '%s' — continuing.", name
                )


if __name__ == "__main__":
    main()
