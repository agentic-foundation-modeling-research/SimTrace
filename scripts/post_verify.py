"""Standalone "post" verifier for already-generated data.

Runs the trajectory-conditioned post verifier over every run in a generated
``runs/`` directory, without regenerating anything (no browser/Playwright). For
each run it reads the reference trajectory + intent from ``basic_info.json`` and
the executed actions from ``action_trace.json``, calls
``TrajAgent.post_verify(...)``, and writes ``post_verify_result.json`` into the
run dir. The output format is identical to the inline generation path, so
``eval.run_eval._compute_post_verify_exclusions`` consumes it unchanged.

Usage:
  python -m scripts.post_verify \\
      --data_dir results_toy_0701/.../1/runs \\
      [--config conf/base.yaml] \\
      [--model googlevertexai-global:gemini-3-flash-preview] \\
      [--force] \\
      [--concurrency 8]

``--data_dir`` may be a ``runs/`` directory or a dir that directly contains one.
Runs that already have ``post_verify_result.json`` are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import types
from pathlib import Path
from typing import Any

import yaml

from src.data_gen.agent import context
from src.data_gen.agent.agent import TrajAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ``load_config`` / ``deep_merge`` are inlined (rather than imported from
# main.run) so this script does not pull in the Playwright/browser import chain.
def load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, val in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def build_llm_config(config_file: str, model_override: str | None):
    """Build the verifier ``llm_config`` from base.yaml, mirroring the block in
    ``main.run.run_generation`` (verifier_llm merged over the main llm block)."""
    base_cfg = load_config(config_file)
    llm_dict = dict(base_cfg.get("llm", {}) or {})
    if model_override:
        llm_dict = deep_merge(llm_dict, {"model": model_override})

    llm_config = types.SimpleNamespace(**llm_dict)
    verifier_overrides = llm_dict.get("verifier_llm")
    if verifier_overrides:
        merged = {**llm_dict, **verifier_overrides}
        merged.pop("verifier_llm", None)
        llm_config.verifier_llm = types.SimpleNamespace(**merged)
    else:
        llm_config.verifier_llm = None
    return llm_config


def resolve_runs_dir(data_dir: Path) -> Path:
    """Accept either a ``runs/`` dir or a dir that directly contains one."""
    if (data_dir / "runs").is_dir():
        return data_dir / "runs"
    return data_dir


def discover_runs(runs_dir: Path, force: bool) -> list[Path]:
    """Immediate child run dirs holding both basic_info.json and
    action_trace.json. Skips already-verified runs unless ``force``."""
    runs: list[Path] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (run_dir / "basic_info.json").is_file():
            continue
        if not (run_dir / "action_trace.json").is_file():
            continue
        if not force and (run_dir / "post_verify_result.json").is_file():
            logger.info("skip (already verified): %s", run_dir.name)
            continue
        runs.append(run_dir)
    return runs


async def verify_run(run_dir: Path, llm_config, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        # ContextVars are copied per task, so setting run_path here keeps
        # LogApiCall's api_trace writes isolated to this run dir.
        (run_dir / "api_trace").mkdir(parents=True, exist_ok=True)
        context.run_path.set(run_dir)

        info = json.loads((run_dir / "basic_info.json").read_text(encoding="utf-8"))
        intent = info.get("intent", "")
        traj_str = info.get("traj", "")
        persona = info.get("persona")
        trajectory_list = traj_str.split(",") if traj_str else []

        raw_actions = json.loads(
            (run_dir / "action_trace.json").read_text(encoding="utf-8")
        )
        # action_trace.json is a list of JSON strings (see experiment.py).
        action_trace_objects = [
            json.loads(a) if isinstance(a, str) else a for a in raw_actions
        ]
        if not action_trace_objects:
            logger.warning("empty action_trace, skipping: %s", run_dir.name)
            return None

        agent = TrajAgent(
            trajectory_list, intent, llm_config=llm_config, persona=persona
        )
        result = await agent.post_verify(trajectory_list, action_trace_objects)

        with open(run_dir / "post_verify_result.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info("verified: %s", run_dir.name)
        return result


def _mean_scores(results: list[dict]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        for key, val in r.items():
            if isinstance(val, dict) and isinstance(val.get("score"), (int, float)):
                sums[key] = sums.get(key, 0.0) + val["score"]
                counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


async def run(data_dir: Path, config: str, model: str | None, force: bool, concurrency: int) -> None:
    runs_dir = resolve_runs_dir(data_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"No runs directory found at {runs_dir}")

    llm_config = build_llm_config(config, model)
    run_dirs = discover_runs(runs_dir, force)
    logger.info("Found %d run(s) to verify under %s", len(run_dirs), runs_dir)
    if not run_dirs:
        return

    sem = asyncio.Semaphore(concurrency)
    tasks = [verify_run(rd, llm_config, sem) for rd in run_dirs]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[dict] = []
    verified = failed = 0
    for run_dir, outcome in zip(run_dirs, outcomes):
        if isinstance(outcome, Exception):
            failed += 1
            logger.error("FAILED %s: %r", run_dir.name, outcome)
        elif outcome is None:
            failed += 1
        else:
            verified += 1
            results.append(outcome)

    logger.info("Done: %d verified, %d failed/skipped-empty", verified, failed)
    means = _mean_scores(results)
    if means:
        logger.info(
            "Mean scores: %s",
            ", ".join(f"{k}={v:.3f}" for k, v in sorted(means.items())),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="A runs/ dir or a dir containing one")
    parser.add_argument("--config", default="conf/base.yaml")
    parser.add_argument("--model", default=None, help="Override llm.model")
    parser.add_argument("--force", action="store_true", help="Re-verify already-verified runs")
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run(
            Path(args.data_dir),
            args.config,
            args.model,
            args.force,
            args.concurrency,
        )
    )
