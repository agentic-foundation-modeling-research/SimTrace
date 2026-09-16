import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .experiment import experiment_async

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict. Nested dicts are
    merged key-by-key; any non-dict value (or a dict overriding a non-dict) replaces the
    base value. `base` is not mutated."""
    result = dict(base)
    for key, val in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _sanitize(text: str) -> str:
    """Make a string safe to use as a single path segment."""
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(text)).strip("-")


def parse_sessions_index(spec) -> list[int]:
    """Parse an index-spec like "0-299" or "0,1,2,3,10-15" into a list of
    0-based indices. Comma separates items; "a-b" is an inclusive range.
    Order is preserved; duplicates are removed keeping first occurrence."""
    indices: list[int] = []
    seen: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            rng = range(start, end + 1)
        else:
            rng = [int(part)]
        for i in rng:
            if i not in seen:
                seen.add(i)
                indices.append(i)
    return indices


def build_config_slug(
    agent_mode: str, base_cfg: dict[str, Any], run_cfg: dict[str, Any]
) -> str:
    """Build a readable, filesystem-safe slug from the distinguishing config knobs so
    that different configurations land in different output folders instead of
    overwriting each other.

    Examples:
        gemini-3-flash-preview__fee1_per0_pla1_ver1__n50
        gpt-5__np8
    """
    raw_model = base_cfg.get("llm", {}).get("model", "unknown")
    model = _sanitize(raw_model.split(":")[-1])
    parts = [model]

    modules = base_cfg.get("simulation_config", {}).get(agent_mode, {}) or {}
    if isinstance(modules, dict) and modules:
        mod_str = "_".join(
            f"{k[:3]}{int(bool(v))}" if isinstance(v, bool) else f"{k[:3]}{v}" for k, v in sorted(modules.items())
        )
        parts.append(mod_str)

    spec = run_cfg.get("sessions_index")
    if spec is not None:
        parts.append(f"n{len(parse_sessions_index(spec))}")

    return "__".join(parts)


def save_config_snapshot(
    base_output_dir: str, base_cfg: dict[str, Any], run_cfg: dict[str, Any]
) -> None:
    """Dump the exact config used for this run to <base_output_dir>/config_used.yaml."""
    snapshot = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base": base_cfg,
        "run_config": run_cfg,
    }
    path = os.path.join(base_output_dir, "config_used.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)


def append_run_registry(registry_path: str, entry: dict[str, Any]) -> None:
    """Append one JSON line describing this run to the central runs_index.jsonl."""
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _safe_ping(cb: Optional[Callable[[dict], None]], evt: dict):
    if not cb:
        return
    try:
        cb(evt)
    except Exception:
        # never let UI progress kill the job
        pass


def _scan_run_status(
    output_dir: str,
) -> tuple[set[str], dict[str, "Path"]]:
    """Scan <output_dir>/runs/ and classify each subfolder as complete or incomplete.

    Complete = basic_info.json + session_data.json present AND no error.txt.
    Returns (completed_ids, incomplete_folders) where incomplete_folders maps
    session_id (or folder name for entries without a session_id) -> Path.
    """
    runs_dir = Path(output_dir) / "runs"
    completed_ids: set[str] = set()
    incomplete_folders: dict[str, Path] = {}

    if not runs_dir.exists():
        return completed_ids, incomplete_folders

    for folder in runs_dir.iterdir():
        if not folder.is_dir():
            continue
        basic_info_path = folder / "basic_info.json"
        session_data_path = folder / "session_data.json"
        error_path = folder / "error.txt"

        session_id = folder.name  # fallback key
        if basic_info_path.exists():
            try:
                with open(basic_info_path) as f:
                    info = json.load(f)
                session_id = info.get("session_id", folder.name) or folder.name
            except Exception:
                pass

        if session_data_path.exists() and not error_path.exists():
            completed_ids.add(session_id)
        else:
            incomplete_folders[session_id] = folder

    return completed_ids, incomplete_folders


def resolve_output_dir(output_dir: str, do_continue: bool) -> bool:
    """Ensure output_dir exists and resolve conflicts with a prior run.

    If a prior run is detected (sessions_input.jsonl present) and do_continue
    is False, the user is prompted to choose:
      [c] Continue — skip completed sessions and run the rest
      [o] Overwrite — wipe the directory and start fresh

    Returns the effective do_continue value to use for this output_dir.
    """
    sessions_input_path = Path(output_dir) / "sessions_input.jsonl"
    prior_run_exists = Path(output_dir).exists() and sessions_input_path.exists()

    if prior_run_exists and not do_continue:
        print(f"\n{'=' * 60}")
        print(f"Prior run detected in: {output_dir}")
        print("  [c] Continue — skip already-completed sessions, run the rest")
        print("  [o] Overwrite — delete all previous results and start fresh")
        print("=" * 60)
        choice = input("Your choice [c/o]: ").strip().lower()
        if choice == "c":
            return True
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            return False

    os.makedirs(output_dir, exist_ok=True)
    return do_continue


def get_continuation_sessions(output_dir: str) -> list[dict]:
    """Read sessions_input.jsonl, delete incomplete run folders, print a summary,
    prompt the user to confirm, and return the sessions still to be processed."""
    sessions_input_path = Path(output_dir) / "sessions_input.jsonl"
    if not sessions_input_path.exists():
        print(f"ERROR: sessions_input.jsonl not found in {output_dir}")
        print("Cannot continue — no prior run record exists for this output dir.")
        sys.exit(1)

    with open(sessions_input_path) as f:
        all_sessions = [json.loads(line) for line in f if line.strip()]

    completed_ids, incomplete_folders = _scan_run_status(output_dir)

    remaining = [
        s for s in all_sessions if s.get("session_id", "") not in completed_ids
    ]

    print(f"\n{'=' * 60}")
    print(f"Continue from : {output_dir}")
    print(f"Total planned : {len(all_sessions)}")
    print(f"Already done  : {len(completed_ids)}")
    print(f"Remaining     : {len(remaining)}")
    if remaining:
        print("Remaining session IDs:")
        for s in remaining:
            print(f"  {s.get('session_id', 'unknown')}")
    print("=" * 60)

    confirm = input("Continue with these sessions? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # Delete incomplete folders so they don't pollute the next run
    for folder in incomplete_folders.values():
        shutil.rmtree(folder, ignore_errors=True)

    return remaining


def _check_and_offer_redo(output_dir: str, expected_ids: list[str]) -> list[dict]:
    """Check run completion after experiment_async finishes.

    Prints a status summary, and if there are incomplete sessions prompts the
    user to redo them.  Returns the list of session dicts to redo (empty if
    all done or user declines).
    """
    completed_ids, incomplete_folders = _scan_run_status(output_dir)
    incomplete_ids = set(expected_ids) - completed_ids

    if not incomplete_ids:
        print(f"\nAll {len(expected_ids)} session(s) completed successfully.")
        return []

    print(
        f"\nRun finished. {len(completed_ids)}/{len(expected_ids)} session(s) succeeded."
    )
    print(f"Incomplete sessions ({len(incomplete_ids)}):")
    for sid in sorted(incomplete_ids):
        print(f"  {sid}")

    # confirm = input("Redo incomplete sessions? [y/N]: ").strip().lower()
    # if confirm != "y":
    #     return []

    # Delete the incomplete run folders before retrying
    for sid in incomplete_ids:
        if sid in incomplete_folders:
            shutil.rmtree(incomplete_folders[sid], ignore_errors=True)

    # Read sessions_input.jsonl to recover full session data for redo
    sessions_input_path = Path(output_dir) / "sessions_input.jsonl"
    if not sessions_input_path.exists():
        print(
            "WARNING: sessions_input.jsonl not found; cannot recover session data for redo."
        )
        return []

    with open(sessions_input_path) as f:
        all_sessions = {
            s.get("session_id", ""): s
            for line in f
            if line.strip()
            for s in [json.loads(line)]
        }

    return [all_sessions[sid] for sid in incomplete_ids if sid in all_sessions]


async def run_action_async(
    sessions_index: str,
    input_filepath: str,
    start_url: str,
    concurrency: int = 4,
    headless: bool = False,
    llm_config=None,
    output_dir: str | None = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    sessions_override: list[dict] | None = None,
    modules=None,
    agent_mode: str | None = None,
):
    def ping(e):
        _safe_ping(on_progress, e)

    if sessions_override is not None:
        # Continuing from a prior run — skip load/sample and don't overwrite sessions_input.jsonl
        sampled = sessions_override
    else:
        with open(input_filepath, "r") as f:
            all_sessions = [json.loads(line) for line in f if line.strip()]

        indices = parse_sessions_index(sessions_index)
        in_range = [i for i in indices if 0 <= i < len(all_sessions)]
        out_of_range = [i for i in indices if not (0 <= i < len(all_sessions))]
        if out_of_range:
            log.warning(
                f"sessions_index requested {len(out_of_range)} out-of-range "
                f"indices (input has {len(all_sessions)} sessions): {out_of_range}"
            )
        sampled = [all_sessions[i] for i in in_range]
        log.info(
            f"Loaded {len(all_sessions)} sessions from {input_filepath}, "
            f"selected {len(sampled)} via sessions_index"
        )

        # save the sessions data use for the data generation into the output dir for record keeping
        path = os.path.join(
            output_dir if output_dir else "./data",
            "sessions_input.jsonl",
        )
        with open(path, "w") as f:
            for s in sampled:
                f.write(json.dumps(s) + "\n")

    agents = [
        {
            "traj": ",".join(s["trajectory"]),
            "intent": s.get("intent", ""),
            "session_id": s.get("session_id", ""),
            "persona": s.get("persona", ""),
            "persona_id": s.get("persona_id", ""),
        }
        for s in sampled
    ]

    total = len(agents)
    ping({"phase": "agents", "status": "start", "total": total})

    results, combined_path = await experiment_async(
        agents=agents,
        start_url=start_url,
        concurrency=concurrency,
        headless=headless,
        llm_config=llm_config,
        output_dir=output_dir,
        on_progress=lambda k, n: ping(
            {"phase": "agents", "status": "progress", "current": k, "total": n}
        ),
        modules=modules,
        agent_mode=agent_mode,
    )

    ping({"phase": "agents", "status": "progress", "current": total, "total": total})

    # Post-run completeness check and optional redo loop
    expected_ids = [s.get("session_id", "") for s in sampled if s.get("session_id")]
    if expected_ids and output_dir:
        redo_sessions = _check_and_offer_redo(output_dir, expected_ids)
        max_redo_attempts = 5
        redo_attempt = 0
        while redo_sessions and redo_attempt < max_redo_attempts:
            redo_attempt += 1
            redo_agents = [
                {
                    "traj": ",".join(s["trajectory"]),
                    "intent": s.get("intent", ""),
                    "session_id": s.get("session_id", ""),
                    "persona": s.get("persona", ""),
                    "persona_id": s.get("persona_id", ""),
                }
                for s in redo_sessions
            ]
            _, combined_path = await experiment_async(
                agents=redo_agents,
                start_url=start_url,
                concurrency=concurrency,
                headless=headless,
                llm_config=llm_config,
                output_dir=output_dir,
                modules=modules,
                agent_mode=agent_mode,
            )
            redo_ids = [
                s.get("session_id", "") for s in redo_sessions if s.get("session_id")
            ]
            redo_sessions = _check_and_offer_redo(output_dir, redo_ids)

    return results, combined_path


def _reset_litellm_logging_worker() -> None:
    """litellm's module-global ``GLOBAL_LOGGING_WORKER`` lazily creates an asyncio.Queue
    bound to whatever event loop is running the first time it logs. When we drive multiple
    experiments in one process, each ``asyncio.run`` opens a fresh loop, and the worker's
    stale queue/task raise ``RuntimeError: <Queue ...> is bound to a different event loop``.
    Dropping the queue and task forces the worker to re-bind to the new loop. Best-effort:
    litellm internals may change, so swallow any failure."""
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        GLOBAL_LOGGING_WORKER._queue = None
        GLOBAL_LOGGING_WORKER._worker_task = None
    except Exception:
        pass


def run_generation(
    do_continue: bool = False,
    *,
    base_cfg: dict[str, Any] | None = None,
    run_cfg: dict[str, Any] | None = None,
    run_index: int = 1,
) -> "Path | None":
    """Drive generation end-to-end, returning the path to the combined session_data JSON
    (or None if no records were produced).

    `run_index` (1-based) nests the output under `<slug>/<run_index>/` so multiple runs
    of the same setting are kept side by side."""
    import types

    # Each experiment runs its own asyncio.run (new event loop). Reset litellm's global
    # logging worker so its internal queue re-binds to this loop instead of a closed one.
    _reset_litellm_logging_worker()

    # Same problem for WebAgentEnv's process-wide Playwright lock/instance: an asyncio.Lock
    # binds to the loop it was first used on, so a stale one survives into the next
    # experiment's fresh loop and raises 'bound to a different event loop'. Drop it here.
    from ..executor.env import WebAgentEnv

    WebAgentEnv._reset_shared_state()

    base_cfg = base_cfg if base_cfg is not None else load_config("conf/base.yaml")
    agent_mode = base_cfg.get("simulation_config", {}).get("agent_mode")
    if agent_mode is None:
        raise ValueError(
            "agent_mode must be specified under simulation_config.agent_mode"
        )
    headless = (
        base_cfg.get("environment", {})
        .get("browser", {})
        .get("launch_options", {})
        .get("headless", True)
    )
    llm_config = types.SimpleNamespace(**base_cfg.get("llm", {}))
    # Optional dedicated model for the verifier (pre + post in traj_cond). Merge the
    # `verifier_llm` overrides over the main llm block so the resulting namespace always
    # carries base_url/request_timeout etc., even when only `model` is overridden. When
    # absent, the verifier falls back to the main llm_config (see TrajAgent.verify).
    _llm_dict = base_cfg.get("llm", {})
    _verifier_overrides = _llm_dict.get("verifier_llm")
    if _verifier_overrides:
        _merged = {**_llm_dict, **_verifier_overrides}
        _merged.pop("verifier_llm", None)
        llm_config.verifier_llm = types.SimpleNamespace(**_merged)
    else:
        llm_config.verifier_llm = None


    if agent_mode not in (
        "traj_cond",
        "persona",
    ):
        raise ValueError(f"Unknown agent_mode: {agent_mode}")

    # Encode the distinguishing config into the folder path so different configs land in
    # different folders instead of silently overwriting one another.
    top_output_dir = base_cfg.get("simulation_config", {}).get(
        "output_dir", "./results"
    )
    slug = build_config_slug(agent_mode, base_cfg, run_cfg)
    # Always nest one level deeper under a run index so repeated runs of the same
    # setting land in sibling folders (<slug>/1/, <slug>/2/, …) instead of clobbering.
    base_output_dir = os.path.join(top_output_dir, agent_mode, slug, str(run_index))

    do_continue = resolve_output_dir(base_output_dir, do_continue)

    # Record exactly what produced this folder: a per-folder snapshot + a central log.
    save_config_snapshot(base_output_dir, base_cfg, run_cfg)
    modules_cfg = base_cfg.get("simulation_config", {}).get(agent_mode, {}) or {}
    # Build the registry entry now (timestamp = start), but defer the append until after
    # generation so we can annotate it with the measured wallclock `elapsed_seconds`.
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_mode": agent_mode,
        "slug": slug,
        "folder": base_output_dir,
        "llm": {
            "model": base_cfg.get("llm", {}).get("model"),
            "provider": base_cfg.get("llm", {}).get("provider"),
            "base_url": base_cfg.get("llm", {}).get("base_url"),
        },
        "modules": modules_cfg,
        "run_config": {
            "sessions_index": run_cfg.get("sessions_index"),
            "total_personas": run_cfg.get("total_personas"),
            "input_filepath": run_cfg.get("input_filepath"),
            "start_url": run_cfg.get("start_url"),
        },
    }

    cfg = run_cfg
    start = time.time()
    sessions_override = (
        get_continuation_sessions(base_output_dir) if do_continue else None
    )
    modules_dict = base_cfg.get("simulation_config", {}).get(agent_mode, {})
    modules = types.SimpleNamespace(**modules_dict) if modules_dict else None
    _, combined_path = asyncio.run(
        run_action_async(
            sessions_index=cfg["sessions_index"],
            input_filepath=cfg["input_filepath"],
            start_url=cfg["start_url"],
            concurrency=cfg.get("concurrency", 4),
            headless=headless,
            llm_config=llm_config,
            output_dir=base_output_dir,
            sessions_override=sessions_override,
            modules=modules,
            agent_mode=agent_mode,
        )
    )

    entry["elapsed_seconds"] = round(time.time() - start, 1)
    append_run_registry(os.path.join(top_output_dir, "runs_index.jsonl"), entry)
    return combined_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Resume from a previous run, skipping already-completed sessions.",
    )
    args = parser.parse_args()
    run_generation(do_continue=args.do_continue)
