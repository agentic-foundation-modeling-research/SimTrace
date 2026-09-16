#!/usr/bin/env python3
"""Convert absolute paths in session_data.json files to relative paths.

Usage:
    python scripts/migrate_session_paths.py <results_dir>

For per-run files (runs/{run_id}/session_data.json), paths become relative to
the run directory (e.g. screenshot/screenshot_0.png).

For aggregated files (session_data_*.json at output_dir root), paths become
relative to the output_dir (e.g. runs/{run_id}/screenshot/screenshot_0.png).

Works even when the paths were written on a different machine (different
absolute prefix), by locating the run_id within the path string.
"""

import json
import sys
from pathlib import Path

PATH_FIELDS = [
    "screenshot_path",
    "dom_snapshot_raw",
    "dom_snapshot_simplified",
    "axtree_snapshot",
]


def _to_relative_per_run(path_str: str, run_id: str) -> str:
    """Return path relative to the run directory, stripping everything up to
    and including the run_id component."""
    p = Path(path_str)
    # Already relative — nothing to do.
    if not p.is_absolute():
        return path_str
    # Try standard relative_to first (same machine).
    # Fall back to string splitting on the run_id.
    parts = p.parts
    try:
        idx = parts.index(run_id)
        return str(Path(*parts[idx + 1:]))
    except ValueError:
        return path_str


def _to_relative_aggregated(path_str: str) -> str:
    """Return path relative to output_dir (i.e. starting with runs/{run_id}/...)."""
    p = Path(path_str)
    if not p.is_absolute():
        return path_str
    # Find the 'runs' component and keep everything from there.
    parts = p.parts
    try:
        idx = parts.index("runs")
        return str(Path(*parts[idx:]))
    except ValueError:
        return path_str


def migrate(results_root: Path) -> None:
    # Per-run session_data.json files inside runs/{run_id}/
    for session_file in results_root.rglob("session_data.json"):
        run_dir = session_file.parent
        run_id = run_dir.name
        records = json.loads(session_file.read_text(encoding="utf-8"))
        changed = False
        for record in records:
            for field in PATH_FIELDS:
                if record.get(field):
                    new = _to_relative_per_run(record[field], run_id)
                    if new != record[field]:
                        record[field] = new
                        changed = True
            if "llm_call" in record:
                new_calls = [_to_relative_per_run(p, run_id) for p in record["llm_call"]]
                if new_calls != record["llm_call"]:
                    record["llm_call"] = new_calls
                    changed = True
        if changed:
            session_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"Updated (per-run):    {session_file}")
        else:
            print(f"No change needed:     {session_file}")

    # Aggregated session_data_{ts}.json files at the output_dir root
    for agg_file in results_root.glob("session_data_*.json"):
        records = json.loads(agg_file.read_text(encoding="utf-8"))
        changed = False
        for record in records:
            for field in PATH_FIELDS:
                if record.get(field):
                    new = _to_relative_aggregated(record[field])
                    if new != record[field]:
                        record[field] = new
                        changed = True
            if "llm_call" in record:
                new_calls = [_to_relative_aggregated(p) for p in record["llm_call"]]
                if new_calls != record["llm_call"]:
                    record["llm_call"] = new_calls
                    changed = True
        if changed:
            agg_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"Updated (aggregated): {agg_file}")
        else:
            print(f"No change needed:     {agg_file}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        sys.exit(1)
    migrate(target)
