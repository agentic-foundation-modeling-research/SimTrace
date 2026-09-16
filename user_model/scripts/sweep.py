"""Sequential hyperparameter-sweep driver for the buyer-sim SFT trainer.

Runs an explicit list of training configs back-to-back, unattended, then
aggregates a ranked summary so you can review all results after the sweep
finishes. Each run gets a fully-merged config YAML (base + per-run overrides),
so the trainer (`user_model.sft.trainer`) needs no new flags — the sweep lives
entirely outside it.

Quick start
-----------
# Dry run — print the launch command + merged config for each run, launch nothing:
python -m user_model.sft.sweep --dry-run

# Run the whole sweep defined in user_model/sft/sweep_config.yaml:
python -m user_model.sft.sweep

# Run a subset by name:
python -m user_model.sft.sweep --only lr2e4_r16,lr1e4_r32

# Re-run everything even if some runs already completed:
python -m user_model.sft.sweep --force

A run is considered "already completed" if <output_dir>/trainer_state.json
exists; such runs are skipped (unless --force), so you can safely re-invoke to
resume a partially-finished sweep.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Repo root = parent of the user_model package.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Eval metrics pulled from the last eval entry of trainer_state.json's log_history.
_EVAL_METRIC_KEYS = ("eval_action_f1", "eval_action_acc", "eval_exact_match_acc", "eval_loss")


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_config(base: dict, overrides: dict, output_dir: str, run_name: str) -> dict:
    """Base config + per-run overrides, with output_dir / wandb_run_name pinned to the run."""
    merged = copy.deepcopy(base)
    merged.update(overrides)
    merged["output_dir"] = output_dir
    merged["wandb_run_name"] = run_name
    return merged


def _build_command(accelerate_config: Path, run_config: Path) -> list[str]:
    return [
        "accelerate", "launch",
        "--config_file", str(accelerate_config),
        "-m", "user_model.sft.trainer",
        "--config", str(run_config),
    ]


def _launch_env_overrides(launch: dict) -> dict[str, str]:
    """The env vars this sweep sets on top of the inherited environment (for display + subprocess)."""
    overrides: dict[str, str] = {}
    cuda = launch.get("cuda_visible_devices")
    if cuda is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = str(cuda)
    for k, v in (launch.get("env") or {}).items():
        overrides[k] = str(v)
    return overrides


def _build_env(launch: dict) -> dict[str, str]:
    # Inherit the full environment, then apply the sweep's explicit overrides.
    return {**os.environ, **_launch_env_overrides(launch)}


def _run_training(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    """Run one training subprocess, streaming stdout/stderr to console and tee-ing to log_path."""
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return proc.wait()


def _read_metrics(output_dir: Path) -> dict[str, Any]:
    """Extract best_metric + final eval metrics from <output_dir>/trainer_state.json."""
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s", state_path, e)
        return {}
    metrics: dict[str, Any] = {
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
    }
    # The last log_history entry carrying eval metrics is the final evaluation.
    for entry in reversed(state.get("log_history", [])):
        if any(k in entry for k in _EVAL_METRIC_KEYS):
            for k in _EVAL_METRIC_KEYS:
                if k in entry:
                    metrics[k] = entry[k]
            break
    return metrics


def _write_summary(output_root: Path, rows: list[dict], greater_is_better: bool) -> None:
    """Write summary.json (ranked by best_metric) and print a table."""
    def sort_key(r: dict):
        bm = r.get("best_metric")
        # Runs without a metric (failed/skipped-incomplete) sort to the bottom either way.
        if bm is None:
            return float("-inf") if greater_is_better else float("inf")
        return bm

    ranked = sorted(rows, key=sort_key, reverse=greater_is_better)

    (output_root / "summary.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )

    logger.info("=== Sweep summary (ranked by best_metric) ===")
    header = f"{'run':<24} {'status':<9} {'best_metric':>12} {'eval_action_f1':>15}"
    logger.info(header)
    for r in ranked:
        bm = r.get("best_metric")
        f1 = r.get("eval_action_f1")
        logger.info(
            "%-24s %-9s %12s %15s",
            r["name"], r["status"],
            f"{bm:.4f}" if isinstance(bm, (int, float)) else "-",
            f"{f1:.4f}" if isinstance(f1, (int, float)) else "-",
        )
    logger.info("Wrote %s / summary.json", output_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential hyperparameter sweep driver.")
    parser.add_argument("--sweep", default="user_model/sft/sweep_config.yaml",
                        help="Path to the sweep spec YAML.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write merged configs and print launch commands, but launch nothing.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run names to run (subset of the spec).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even runs whose trainer_state.json already exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    spec = _load_yaml(Path(args.sweep))
    base_config = _load_yaml(REPO_ROOT / spec["base_config"])
    output_root = Path(spec["output_root"])
    launch = spec.get("launch", {})
    accelerate_config = REPO_ROOT / launch.get("accelerate_config", "user_model/sft/accelerate_config.yaml")
    env = _build_env(launch)
    env_prefix = " ".join(f"{k}={v}" for k, v in _launch_env_overrides(launch).items())
    greater_is_better = bool(base_config.get("greater_is_better", True))

    runs = spec["runs"]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        runs = [r for r in runs if r["name"] in wanted]
        missing = wanted - {r["name"] for r in runs}
        if missing:
            raise SystemExit(f"--only names not found in sweep spec: {sorted(missing)}")
    if not runs:
        raise SystemExit("No runs selected.")

    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Sweep: %d run(s) -> %s", len(runs), output_root)

    summary_rows: list[dict] = []
    for i, run in enumerate(runs, 1):
        name = run["name"]
        overrides = run.get("overrides", {}) or {}
        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)

        merged = _merge_config(base_config, overrides, str(output_dir), name)
        run_config = output_dir / "config.yaml"
        with open(run_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)

        cmd = _build_command(accelerate_config, run_config)
        row: dict[str, Any] = {"name": name, "overrides": overrides}

        logger.info("[%d/%d] %s", i, len(runs), name)
        logger.info("  overrides: %s", json.dumps(overrides))
        logger.info("  config:    %s", run_config)
        logger.info("  command:   %s %s", env_prefix, " ".join(cmd))

        if args.dry_run:
            row["status"] = "dry-run"
            summary_rows.append(row)
            continue

        state_path = output_dir / "trainer_state.json"
        if state_path.exists() and not args.force:
            logger.info("  -> already completed (trainer_state.json exists); skipping. Use --force to re-run.")
            row["status"] = "skipped"
            row.update(_read_metrics(output_dir))
            summary_rows.append(row)
            continue

        rc = _run_training(cmd, env, output_dir / "train.log")
        row["status"] = "done" if rc == 0 else f"failed(rc={rc})"
        if rc != 0:
            logger.warning("  -> run %s exited with code %d; continuing to next run.", name, rc)
        row.update(_read_metrics(output_dir))
        summary_rows.append(row)
        # Refresh the summary after every run so partial results survive a crash/interrupt.
        _write_summary(output_root, summary_rows, greater_is_better)

    if not args.dry_run:
        _write_summary(output_root, summary_rows, greater_is_better)
    else:
        logger.info("Dry run complete — %d merged config(s) written under %s", len(summary_rows), output_root)


if __name__ == "__main__":
    main()
