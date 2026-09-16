"""Build and publish pseudonymized shopping data with base64 image payloads.

The output contains three Hugging Face dataset configurations with shared
train/test assignments:

    action: store_id, session_id, user_id, timestamp, action_type, target,
            rationale, input_text, simplified_dom, action_json, image
    image:  image_id, image
    user:   store_id, session_id, user_id, persona, intent

The action table intentionally excludes clicked_url and raw_dom.
Image payloads are stored as plain RFC 4648 base64 strings rather than local
paths or datasets.Image values. Sessions containing move, key_press, hover,
or goto_url actions are dropped before sampling, splitting, statistics, and
publishing.

The three identifier columns contain deterministic, keyed UUID pseudonyms.
Set BUYER_SIM_PSEUDONYMIZATION_KEY to a secret containing at least 32 bytes;
the key is never written to output.

Example:

    export BUYER_SIM_PSEUDONYMIZATION_KEY="$(openssl rand -hex 32)"
    python -m user_model.publish_data_to_hf \
        --config conf/publish_data_to_hf.example.yaml \
        --push_to_hub
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import statistics
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from user_model.prompt_templates import normalize_assistant_payload

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CONFIG_NAMES = ("action", "image", "user")
PSEUDONYMIZATION_KEY_ENV = "BUYER_SIM_PSEUDONYMIZATION_KEY"
MIN_PSEUDONYMIZATION_KEY_BYTES = 32
SPLIT_NAMES = ("train", "test")
CONFIG_SPLITS = {
    "action": SPLIT_NAMES,
    "image": SPLIT_NAMES,
    "user": SPLIT_NAMES,
}
EXCLUDED_ACTION_TYPES = frozenset({"move", "key_press", "hover", "goto_url"})


@dataclass(frozen=True)
class PreparedRun:
    """All publishable rows originating from one simulation run."""

    run_dir: Path
    split_group: str
    action_rows: list[dict]
    image_rows: list[dict]
    user_row: dict


def _json_object(value: object) -> dict:
    """Return ``value`` as an object, accepting a JSON-object string."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_string(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _persona_without_buyer_id(raw_persona: object) -> tuple[dict, str | None]:
    persona = _json_object(raw_persona)
    buyer_id = persona.pop("buyer_id", None)
    return persona, str(buyer_id) if buyer_id not in (None, "") else None


def _store_id(info: dict, persona: dict, buyer_id: str | None) -> str | None:
    """Resolve store id from explicit metadata, then the buyer/persona id suffix."""
    explicit = info.get("store_id") or persona.get("store_id")
    if explicit not in (None, ""):
        return str(explicit)
    for candidate in (buyer_id, info.get("persona_id"), info.get("user_id")):
        match = re.search(r"-(\d+)$", str(candidate or ""))
        if match:
            return match.group(1)
    return None


def _user_id(info: dict, buyer_id: str | None) -> str | None:
    value = info.get("user_id") or info.get("persona_id") or buyer_id
    return str(value) if value not in (None, "") else None


def _resolve_file(raw_path: object, run_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = run_dir / path
    return path if path.is_file() else None


def _read_text(raw_path: object, run_dir: Path) -> str | None:
    path = _resolve_file(raw_path, run_dir)
    if path is None:
        if raw_path:
            logger.warning("DOM file not found for %s: %s", run_dir, raw_path)
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _dom_index(record: dict, fallback: int) -> int:
    """Use the DOM filename's trailing integer as the screenshot index."""
    raw = record.get("dom_snapshot_simplified") or record.get("dom_snapshot_raw")
    if raw:
        match = re.search(r"(\d+)$", Path(str(raw)).stem)
        if match:
            return int(match.group(1))
    return fallback


def _screenshot_path(record: dict, run_dir: Path, index: int) -> Path | None:
    """Prefer pre-highlight screenshot N, then fall back to screenshot_N.png."""
    screenshot_dir = run_dir / "screenshot"
    preferred = [
        screenshot_dir / f"screenshot_{index}_full_page_pre_highlight.png",
        screenshot_dir / f"screenshot_{index}_pre_highlight.png",
    ]
    if screenshot_dir.is_dir():
        preferred.extend(sorted(screenshot_dir.glob(f"screenshot_{index}*pre_highlight.png")))
    seen: set[Path] = set()
    for candidate in preferred:
        if candidate not in seen and candidate.is_file():
            return candidate.resolve()
        seen.add(candidate)

    fallback = screenshot_dir / f"screenshot_{index}.png"
    if fallback.is_file():
        return fallback.resolve()

    # Older runs may place an absolute screenshot_path outside the conventional
    # directory.  Only use it when it is for the same DOM/screenshot index.
    recorded = _resolve_file(record.get("screenshot_path"), run_dir)
    if recorded is not None and _dom_index({"dom_snapshot_simplified": recorded}, -1) == index:
        return recorded.resolve()
    return None


def _image_id(path: Path) -> str:
    """Content-address image ids are stable and naturally deduplicate images."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{digest.hexdigest()[:24]}{path.suffix.lower() or '.png'}"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _store_id_from_raw_data(raw_data: Path) -> str | None:
    match = re.search(r"(?:^|_)sample_(\d+)$", raw_data.name)
    return match.group(1) if match else None


def _post_verify_average(run_dir: Path) -> float | None:
    """Return the mean numeric verifier score, or ``None`` when unavailable."""
    result_path = run_dir / "post_verify_result.json"
    if not result_path.is_file():
        return None
    try:
        result = _load_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse %s: %s; keeping run", result_path, exc)
        return None
    if not isinstance(result, dict):
        logger.warning("Expected an object in %s; keeping run", result_path)
        return None
    scores = [
        value["score"]
        for value in result.values()
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float))
    ]
    return sum(scores) / len(scores) if scores else None


def _prepare_run(
    run_dir: Path,
    split_by: str,
    post_verify_threshold: float | None,
) -> PreparedRun | None:
    session_path = run_dir / "session_data.json"
    info_path = run_dir / "basic_info.json"
    if not session_path.is_file() or not info_path.is_file() or (run_dir / "error.txt").exists():
        return None
    if post_verify_threshold is not None:
        score = _post_verify_average(run_dir)
        if score is not None and score < post_verify_threshold:
            logger.info(
                "Dropping %s: post-verify average %.3f < %.3f",
                run_dir,
                score,
                post_verify_threshold,
            )
            return None
    try:
        raw_records = _load_json(session_path)
        info = _load_json(info_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable run %s: %s", run_dir, exc)
        return None
    if not isinstance(raw_records, list) or not isinstance(info, dict):
        logger.warning("Skipping %s: expected session list and basic-info object", run_dir)
        return None

    persona, buyer_id = _persona_without_buyer_id(info.get("persona"))
    store_id = _store_id(info, persona, buyer_id)
    user_id = _user_id(info, buyer_id)
    session_id = str(
        info.get("session_id")
        or next((row.get("session_id") for row in raw_records if isinstance(row, dict)), "")
        or run_dir.name
    )
    if store_id is None:
        logger.warning("No store_id could be resolved for %s", run_dir)
    if user_id is None:
        logger.warning("No user_id/persona_id could be resolved for %s", run_dir)

    images: dict[str, dict] = {}
    actions: list[dict] = []
    for fallback_index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            logger.warning("Skipping non-object action %d in %s", fallback_index, run_dir)
            continue
        raw_action = record.get("synthetic_action", "")
        payload = normalize_assistant_payload(
            json.dumps(raw_action, ensure_ascii=False)
            if isinstance(raw_action, dict)
            else raw_action
        )
        if not payload:
            logger.warning("Skipping invalid synthetic_action %d in %s", fallback_index, run_dir)
            continue
        # Defense in depth: these source-only fields must not be retained even
        # if an atypical synthetic_action payload happens to contain them.
        payload.pop("clicked_url", None)
        payload.pop("raw_dom", None)

        index = _dom_index(record, fallback_index)
        screenshot = _screenshot_path(record, run_dir, index)
        image_id = None
        if screenshot is not None:
            image_id = _image_id(screenshot)
            images.setdefault(
                image_id,
                {"image_id": image_id, "image": str(screenshot)},
            )

        actions.append(
            {
                "store_id": store_id,
                "session_id": str(record.get("session_id") or session_id),
                "user_id": user_id,
                "timestamp": (
                    str(record["timestamp"]) if record.get("timestamp") is not None else None
                ),
                "action_type": (
                    str(payload["action"]) if payload.get("action") is not None else None
                ),
                "target": str(payload["target"]) if payload.get("target") is not None else None,
                "rationale": (
                    str(payload["rationale"]) if payload.get("rationale") is not None else None
                ),
                "input_text": str(payload["text"]) if payload.get("text") is not None else None,
                "simplified_dom": _read_text(record.get("dom_snapshot_simplified"), run_dir),
                "action_json": _json_string(payload),
                "image": image_id,
            }
        )

    if not actions:
        logger.warning("Skipping %s: no valid actions", run_dir)
        return None

    user_row = {
        "store_id": store_id,
        "session_id": session_id,
        "user_id": user_id,
        "persona": _json_string(persona),
        "intent": str(info["intent"]) if info.get("intent") is not None else None,
    }
    group = user_id if split_by == "user" and user_id else session_id
    return PreparedRun(run_dir, str(group), actions, list(images.values()), user_row)


def _iter_run_dirs(run_roots: list[str | Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for raw_root in run_roots:
        root = Path(raw_root)
        if not root.exists():
            raise FileNotFoundError(f"Run directory does not exist: {root}")
        candidates: list[Path]
        if (root / "session_data.json").is_file():
            candidates = [root]
        elif (root / "runs").is_dir():
            candidates = sorted(path for path in (root / "runs").iterdir() if path.is_dir())
        else:
            candidates = sorted(
                path.parent for path in root.rglob("session_data.json") if path.is_file()
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _split_runs(
    runs: list[PreparedRun], test_ratio: float, seed: int
) -> dict[str, list[PreparedRun]]:
    if not 0 <= test_ratio < 1:
        raise ValueError(f"test_ratio must satisfy 0 <= ratio < 1, got {test_ratio}")
    groups: dict[str, list[PreparedRun]] = {}
    for run in runs:
        groups.setdefault(run.split_group, []).append(run)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    if test_ratio == 0 or len(group_ids) < 2:
        n_test = 0
    else:
        n_test = max(1, min(len(group_ids) - 1, round(len(group_ids) * test_ratio)))
    test_groups = set(group_ids[:n_test])
    return {
        "train": [run for run in runs if run.split_group not in test_groups],
        "test": [run for run in runs if run.split_group in test_groups],
    }


def _write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _base64_image(path: Path) -> str:
    """Return the file bytes as an ASCII RFC 4648 base64 string."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _encode_image_jsonl(path: Path) -> None:
    """Atomically replace image paths in a generated JSONL file with base64."""
    temporary_path: Path | None = None
    try:
        with path.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_path = Path(row["image"])
                try:
                    row["image"] = _base64_image(image_path)
                except OSError as exc:
                    raise OSError(
                        f"Could not encode image {image_path} referenced by "
                        f"{path}:{line_number}"
                    ) from exc
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _encode_image_tables(paths: Mapping[str, Mapping[str, Path]]) -> None:
    for path in paths.get("image", {}).values():
        _encode_image_jsonl(path)


STATISTICS_ACTION_FIELDS = (
    "timestamp",
    "target",
    "rationale",
    "input_text",
    "image",
)


def _new_statistics_state() -> dict:
    """Return the compact mutable state used while streaming JSONL rows."""
    return {
        "actions": 0,
        "action_types": Counter(),
        "session_lengths": Counter(),
        "stores": set(),
        "users": set(),
        "user_sessions": {},
        "sessions": set(),
        "images": set(),
        "present_fields": Counter(),
        "timestamp_bounds": {},
    }


def _timestamp_seconds(value: object) -> float | None:
    """Parse an ISO-8601 or numeric timestamp for duration calculations."""
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip()
    try:
        numeric = float(text)
        return numeric if math.isfinite(numeric) else None
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp()


def _update_action_statistics(state: dict, row: dict) -> None:
    store_id = row.get("store_id")
    session_id = row.get("session_id")
    session_key = (store_id, session_id)
    state["actions"] += 1
    state["stores"].add(store_id)
    state["sessions"].add(session_key)
    state["session_lengths"][session_key] += 1
    if row.get("user_id") not in (None, ""):
        state["users"].add(row["user_id"])
        state["user_sessions"].setdefault(row["user_id"], set()).add(session_key)
    if row.get("image") not in (None, ""):
        state["images"].add(row["image"])
    action_type = row.get("action_type")
    action_name = str(action_type) if action_type not in (None, "") else "<missing>"
    state["action_types"][action_name] += 1
    for field in STATISTICS_ACTION_FIELDS:
        if row.get(field) not in (None, ""):
            state["present_fields"][field] += 1

    timestamp = _timestamp_seconds(row.get("timestamp"))
    if timestamp is not None:
        bounds = state["timestamp_bounds"].get(session_key)
        state["timestamp_bounds"][session_key] = (
            timestamp if bounds is None else min(bounds[0], timestamp),
            timestamp if bounds is None else max(bounds[1], timestamp),
        )


def _update_user_statistics(state: dict, row: dict) -> None:
    session_key = (row.get("store_id"), row.get("session_id"))
    state["stores"].add(row.get("store_id"))
    state["sessions"].add(session_key)
    if row.get("user_id") not in (None, ""):
        state["users"].add(row["user_id"])
        state["user_sessions"].setdefault(row["user_id"], set()).add(session_key)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        position - lower
    )


def _distribution(values) -> dict[str, int | float | None]:
    """Summarize a numeric sequence using common dataset-paper statistics."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "average": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "p25": None,
            "p75": None,
            "maximum": None,
        }

    def tidy(value: float) -> int | float:
        rounded = round(value, 2)
        return int(rounded) if rounded.is_integer() else rounded

    return {
        "average": tidy(statistics.fmean(ordered)),
        "median": tidy(statistics.median(ordered)),
        "standard_deviation": tidy(statistics.pstdev(ordered)),
        "minimum": tidy(ordered[0]),
        "p25": tidy(_percentile(ordered, 0.25)),
        "p75": tidy(_percentile(ordered, 0.75)),
        "maximum": tidy(ordered[-1]),
    }


def _finalize_statistics(state: dict, number_of_images: int | None = None) -> dict:
    actions = state["actions"]
    action_types = {
        name: {
            "count": count,
            "percentage": round(100 * count / actions, 2) if actions else 0.0,
        }
        for name, count in sorted(
            state["action_types"].items(), key=lambda item: (-item[1], item[0])
        )
    }
    durations = [
        maximum - minimum
        for minimum, maximum in state["timestamp_bounds"].values()
    ]
    result = {
        "number_of_stores": len(state["stores"]),
        "number_of_sessions": len(state["sessions"]),
        "number_of_users": len(state["users"]),
        "number_of_actions": actions,
        "number_of_images": (
            len(state["images"]) if number_of_images is None else number_of_images
        ),
        "action_types": action_types,
        "session_length_actions": _distribution(state["session_lengths"].values()),
        "sessions_per_user": _distribution(
            len(sessions) for sessions in state["user_sessions"].values()
        ),
        "session_duration_seconds": {
            **_distribution(durations),
            "sessions_with_parseable_timestamps": len(durations),
        },
        "action_field_completeness": {
            field: {
                "present_count": state["present_fields"][field],
                "percentage": (
                    round(100 * state["present_fields"][field] / actions, 2)
                    if actions
                    else 0.0
                ),
            }
            for field in STATISTICS_ACTION_FIELDS
        },
    }
    return result


def _aggregate_dataset_statistics(
    action_splits: Mapping[str, object],
    user_splits: Mapping[str, object],
    image_splits: Mapping[str, object],
) -> dict:
    """Aggregate statistics from one-pass row iterables keyed by split name."""
    overall = _new_statistics_state()
    per_store: dict[str, dict] = {}
    per_split: dict[str, dict] = {}
    image_ids_by_split: dict[str, set] = {}

    for split, rows in action_splits.items():
        split_state = per_split.setdefault(split, _new_statistics_state())
        for row in rows:
            store_key = str(row.get("store_id") or "<missing>")
            store_state = per_store.setdefault(store_key, _new_statistics_state())
            _update_action_statistics(overall, row)
            _update_action_statistics(split_state, row)
            _update_action_statistics(store_state, row)

    for split, rows in user_splits.items():
        split_state = per_split.setdefault(split, _new_statistics_state())
        for row in rows:
            store_key = str(row.get("store_id") or "<missing>")
            store_state = per_store.setdefault(store_key, _new_statistics_state())
            _update_user_statistics(overall, row)
            _update_user_statistics(split_state, row)
            _update_user_statistics(store_state, row)

    all_image_ids: set = set()
    for split, rows in image_splits.items():
        image_ids = image_ids_by_split.setdefault(split, set())
        for row in rows:
            image_id = row.get("image_id")
            if image_id not in (None, ""):
                image_ids.add(image_id)
                all_image_ids.add(image_id)

    overall_result = _finalize_statistics(
        overall, len(all_image_ids) if image_splits else None
    )
    return {
        "overall": overall_result,
        "per_store": {
            store_id: _finalize_statistics(state)
            for store_id, state in sorted(per_store.items())
        },
        "per_split": {
            split: _finalize_statistics(
                state,
                len(image_ids_by_split.get(split, set()))
                if split in image_splits
                else None,
            )
            for split, state in sorted(per_split.items())
        },
    }


def _statistics_for_prepared_runs(
    split_runs: Mapping[str, list[PreparedRun]],
) -> dict:
    """Compute statistics from the exact prepared rows written to each split."""

    def action_rows(runs: list[PreparedRun]) -> Iterator[dict]:
        for run in runs:
            yield from run.action_rows

    def user_rows(runs: list[PreparedRun]) -> Iterator[dict]:
        for run in runs:
            yield run.user_row

    def image_rows(runs: list[PreparedRun]) -> Iterator[dict]:
        for run in runs:
            yield from run.image_rows

    return _aggregate_dataset_statistics(
        {
            split: action_rows(runs)
            for split, runs in split_runs.items()
        },
        {
            split: user_rows(runs)
            for split, runs in split_runs.items()
        },
        {
            split: image_rows(runs)
            for split, runs in split_runs.items()
        },
    )


def _build_from_yaml_for_statistics(
    config_path: str | Path,
    output_path: str | Path | None,
    pseudonymization_key: str | bytes | None = None,
) -> tuple[dict[str, dict[str, Path]], dict, dict]:
    run_dirs, raw_data_dirs, store_ids, config = _load_yaml_config(config_path)
    output_dir = config.get("output_dir", "data/hf_publish_v3")
    paths, statistics_result = build_hf_dataset(
        run_dirs=run_dirs,
        raw_data_dirs=raw_data_dirs,
        store_ids=store_ids,
        output_dir=output_dir,
        test_ratio=config.get("test_ratio", 0.2),
        seed=config.get("seed", 42),
        split_by=config.get("split_by", "user"),
        sample=config.get("sample"),
        post_verify_threshold=config.get("post_verify_threshold"),
        pseudonymization_key=pseudonymization_key,
        return_statistics=True,
        statistics_output_path=output_path,
    )
    return paths, statistics_result, config


def get_dataset_statistics(
    config_path: str | Path,
    output_path: str | Path | None = None,
    *,
    pseudonymization_key: str | bytes | None = None,
) -> dict:
    """Build from YAML and return statistics for the exact publishable rows.

    This uses the same post-verifier filtering, sampling, pseudonymization, and
    split assignment as :func:`build_hf_dataset`. It never reads data back from
    Hugging Face. The configured ``output_dir`` receives the local dataset and
    ``dataset_statistics.json`` unless ``output_path`` overrides that filename.
    """
    _, statistics_result, _ = _build_from_yaml_for_statistics(
        config_path, output_path, pseudonymization_key
    )
    return statistics_result


def publish_from_yaml(
    config_path: str | Path,
    *,
    token: str | bool | None = None,
    pseudonymization_key: str | bytes | None = None,
) -> dict:
    """Build, push the exact built rows, and return their statistics."""
    paths, statistics_result, config = _build_from_yaml_for_statistics(
        config_path, None, pseudonymization_key
    )
    repo_id = config.get("repo_id")
    if not repo_id:
        raise ValueError("YAML config requires repo_id for publishing")
    push_to_hub(
        paths,
        str(repo_id),
        private=config.get("private", True),
        token=token,
    )
    return statistics_result


def _assign_expected_store_id(run: PreparedRun, expected_store_id: str | None) -> None:
    if expected_store_id is None:
        return
    actual = run.user_row.get("store_id")
    if actual is not None and str(actual) != expected_store_id:
        raise ValueError(
            f"Store mismatch: run {run.run_dir} resolves to store_id={actual}, "
            f"but its raw_data/config mapping specifies {expected_store_id}"
        )
    run.user_row["store_id"] = expected_store_id
    for row in run.action_rows:
        row["store_id"] = expected_store_id


def _resolve_pseudonymization_key(value: str | bytes | None = None) -> bytes:
    """Return and validate the key used to pseudonymize identifiers."""
    if value is None:
        value = os.environ.get(PSEUDONYMIZATION_KEY_ENV)
    if value in (None, "", b""):
        raise ValueError(
            f"Set {PSEUDONYMIZATION_KEY_ENV} to a secret containing at least "
            f"{MIN_PSEUDONYMIZATION_KEY_BYTES} bytes"
        )
    key = value if isinstance(value, bytes) else value.encode("utf-8")
    if len(key) < MIN_PSEUDONYMIZATION_KEY_BYTES:
        raise ValueError(
            f"{PSEUDONYMIZATION_KEY_ENV} must contain at least "
            f"{MIN_PSEUDONYMIZATION_KEY_BYTES} bytes"
        )
    return key


def _fake_id(
    id_type: str, original_id: object, pseudonymization_key: bytes
) -> str | None:
    """Return a stable keyed UUID pseudonym for a non-empty source identifier."""
    if original_id in (None, ""):
        return None
    source = f"buyer-sim-gen/hf-v2/{id_type}/{original_id}"
    digest = hmac.digest(pseudonymization_key, source.encode("utf-8"), "sha256")
    return str(uuid.UUID(bytes=digest[:16]))


def _pseudonymize_run(run: PreparedRun, pseudonymization_key: bytes) -> None:
    """Replace source IDs and add the run's fake user ID to every action."""
    original_store_id = run.user_row.get("store_id")
    original_user_id = run.user_row.get("user_id")
    fake_store_id = _fake_id("store", original_store_id, pseudonymization_key)
    fake_user_id = _fake_id("user", original_user_id, pseudonymization_key)

    run.user_row["store_id"] = fake_store_id
    run.user_row["session_id"] = _fake_id(
        "session", run.user_row.get("session_id"), pseudonymization_key
    )
    run.user_row["user_id"] = fake_user_id
    for row in run.action_rows:
        row["store_id"] = fake_store_id
        row["session_id"] = _fake_id(
            "session", row.get("session_id"), pseudonymization_key
        )
        row["user_id"] = fake_user_id


def build_hf_dataset(
    run_dirs: str | Path | list[str | Path],
    output_dir: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
    split_by: str = "user",
    sample: int | None = None,
    post_verify_threshold: float | None = None,
    raw_data_dirs: str | Path | list[str | Path] | None = None,
    store_ids: list[str | int | None] | None = None,
    pseudonymization_key: str | bytes | None = None,
    return_statistics: bool = False,
    statistics_output_path: str | Path | None = None,
) -> dict[str, dict[str, Path]] | tuple[dict[str, dict[str, Path]], dict]:
    """Build local JSONL sources for the action, image, and user configurations.

    Runs below ``post_verify_threshold`` are removed first.  When ``sample`` is
    positive, that many remaining complete simulation runs (sessions) are then
    selected before splitting. Sampling complete runs preserves all joins
    between the action, image, and user configurations. Runs without a numeric
    post-verifier score are retained. Any complete run/session containing an
    action in EXCLUDED_ACTION_TYPES is removed before sampling, splitting,
    statistics, and output generation.
    """
    if split_by not in {"user", "session"}:
        raise ValueError(f"split_by must be 'user' or 'session', got {split_by!r}")
    excluded_actions = set(EXCLUDED_ACTION_TYPES)
    secret_key = _resolve_pseudonymization_key(pseudonymization_key)
    roots = [run_dirs] if isinstance(run_dirs, (str, Path)) else list(run_dirs)
    if raw_data_dirs is None:
        raw_roots: list[Path] | None = None
    else:
        raw_values = (
            [raw_data_dirs]
            if isinstance(raw_data_dirs, (str, Path))
            else list(raw_data_dirs)
        )
        if len(raw_values) != len(roots):
            raise ValueError(
                "raw_data_dirs must contain exactly one folder per run_dir "
                f"({len(roots)} run_dir, {len(raw_values)} raw_data)"
            )
        raw_roots = [Path(value) for value in raw_values]
    if store_ids is not None and len(store_ids) != len(roots):
        raise ValueError("store_ids must contain exactly one value per run_dir")

    expected_store_ids: list[str | None] = []
    for index in range(len(roots)):
        explicit = store_ids[index] if store_ids is not None else None
        inferred = _store_id_from_raw_data(raw_roots[index]) if raw_roots else None
        expected_store_ids.append(str(explicit) if explicit not in (None, "") else inferred)

    runs: list[PreparedRun] = []
    excluded_runs = 0
    for root, expected_store_id in zip(roots, expected_store_ids):
        for run_dir in _iter_run_dirs([root]):
            prepared = _prepare_run(run_dir, split_by, post_verify_threshold)
            if prepared is not None:
                matching_actions = sorted(
                    {
                        str(row.get("action_type") or "").strip().lower()
                        for row in prepared.action_rows
                    }
                    & excluded_actions
                )
                if matching_actions:
                    excluded_runs += 1
                    logger.info(
                        "Dropping %s: contains excluded action type(s): %s",
                        run_dir,
                        ", ".join(matching_actions),
                    )
                    continue
                _assign_expected_store_id(prepared, expected_store_id)
                _pseudonymize_run(prepared, secret_key)
                runs.append(prepared)
    if not runs:
        if excluded_runs:
            raise ValueError(
                "No valid simulation runs remain after excluding action types "
                f"{sorted(excluded_actions)} under: {roots}"
            )
        raise ValueError(f"No valid simulation runs found under: {roots}")

    available_runs = len(runs)
    if sample is not None and sample > 0:
        if sample > available_runs:
            raise ValueError(
                f"Requested sample={sample}, but only {available_runs} valid run(s) "
                f"were found under: {roots}"
            )
        runs = random.Random(seed).sample(runs, sample)
        logger.info("Sampled %d of %d valid run(s) before splitting", sample, available_runs)
    split_runs = _split_runs(runs, test_ratio, seed)
    statistics_result = _statistics_for_prepared_runs(split_runs)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    active_configs = CONFIG_NAMES
    paths: dict[str, dict[str, Path]] = {name: {} for name in active_configs}
    counts: dict[str, dict[str, int]] = {name: {} for name in active_configs}
    for config in active_configs:
        config_dir = output / config
        config_dir.mkdir(parents=True, exist_ok=True)
        for split in CONFIG_SPLITS[config]:
            path = config_dir / f"{split}.jsonl"
            if config == "action":
                selected = split_runs[split]
                rows = (row for run in selected for row in run.action_rows)
            elif config == "image":
                selected = split_runs[split]
                # Deduplicate content-addressed image ids within each split.
                unique = {
                    row["image_id"]: row for run in selected for row in run.image_rows
                }
                rows = iter(unique.values())
            elif config == "user":
                selected = split_runs[split]
                rows = (run.user_row for run in selected)
            else:
                raise ValueError(f"Unknown config: {config}")
            counts[config][split] = _write_jsonl(path, rows)
            paths[config][split] = path

    manifest = {
        "configs": list(active_configs),
        "splits": {config: list(CONFIG_SPLITS[config]) for config in active_configs},
        "split_by": split_by,
        "test_ratio": test_ratio,
        "seed": seed,
        "sample": sample,
        "post_verify_threshold": post_verify_threshold,
        "excluded_action_types": sorted(excluded_actions),
        "excluded_runs": excluded_runs,
        "available_runs": available_runs,
        "sampled_runs": len(runs),
        "runs": {split: len(items) for split, items in split_runs.items()},
        "rows": counts,
        "statistics_file": str(
            Path(statistics_output_path)
            if statistics_output_path is not None
            else output / "dataset_statistics.json"
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    statistics_destination = (
        Path(statistics_output_path)
        if statistics_output_path is not None
        else output / "dataset_statistics.json"
    )
    statistics_destination.parent.mkdir(parents=True, exist_ok=True)
    statistics_destination.write_text(
        json.dumps(statistics_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _encode_image_tables(paths)
    logger.info("Built HF dataset sources in %s: %s", output, counts)
    logger.info("Encoded image table payloads as base64 strings")
    if return_statistics:
        return paths, statistics_result
    return paths


def _hf_features(config: str):
    from datasets import Features, Value

    if config == "action":
        return Features(
            {
                key: Value("string")
                for key in (
                    "store_id",
                    "session_id",
                    "user_id",
                    "timestamp",
                    "action_type",
                    "target",
                    "rationale",
                    "input_text",
                    "simplified_dom",
                    "action_json",
                    "image",
                )
            }
        )
    if config == "image":
        return Features({"image_id": Value("string"), "image": Value("string")})
    if config == "user":
        return Features(
            {
                key: Value("string")
                for key in ("store_id", "session_id", "user_id", "persona", "intent")
            }
        )
    raise ValueError(f"Unknown config: {config}")


def _dataset_for_jsonl(config: str, path: Path):
    """Load only declared fields, preserving base64 images as strings."""
    from datasets import Dataset, load_dataset

    features = _hf_features(config)
    if path.stat().st_size == 0:
        return Dataset.from_dict(
            {field_name: [] for field_name in features},
            features=features,
        )
    dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )
    undeclared_columns = [
        column for column in dataset.column_names if column not in features
    ]
    if undeclared_columns:
        dataset = dataset.remove_columns(undeclared_columns)
    missing_columns = [field for field in features if field not in dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required {config} field(s): "
            f"{', '.join(missing_columns)}"
        )
    return dataset.cast(features)


def push_to_hub(
    paths: dict[str, dict[str, Path]],
    repo_id: str,
    private: bool = True,
    token: str | bool | None = None,
) -> None:
    """Push action/image/user configs with base64 image strings."""
    from datasets import DatasetDict

    for config in paths:
        dataset = DatasetDict(
            {
                split: _dataset_for_jsonl(config, paths[config][split])
                for split in CONFIG_SPLITS[config]
            }
        )
        logger.info(
            "Pushing config %s (%s) to %s",
            config,
            ", ".join(f"{split}={len(rows)}" for split, rows in dataset.items()),
            repo_id,
        )
        push_kwargs = {"config_name": config, "private": private}
        if token is not None:
            push_kwargs["token"] = token
        dataset.push_to_hub(repo_id, **push_kwargs)
    logger.info("Published dataset at https://huggingface.co/datasets/%s", repo_id)


def _expand_store_config(config: dict) -> tuple[list[str], list[str | None]]:
    stores = config.get("stores")
    if not isinstance(stores, list) or not stores:
        raise ValueError("YAML config requires a non-empty 'stores' list")

    run_dirs: list[str] = []
    store_ids: list[str | None] = []
    for index, store in enumerate(stores):
        if not isinstance(store, dict):
            raise ValueError(f"stores[{index}] must be an object")
        raw_data = store.get("raw_data")
        run_values = store.get("run_dir")
        if not run_values:
            raise ValueError(f"stores[{index}] requires run_dir")
        if isinstance(run_values, (str, Path)):
            run_values = [run_values]
        if not isinstance(run_values, list) or not run_values:
            raise ValueError(f"stores[{index}].run_dir must be a path or non-empty list")
        value = store.get("store_id")
        if value in (None, "") and raw_data:
            value = _store_id_from_raw_data(Path(str(raw_data)))
        for run_dir in run_values:
            run_dirs.append(str(run_dir))
            store_ids.append(str(value) if value not in (None, "") else None)
    return run_dirs, store_ids


def _load_yaml_config(
    path: str | Path,
) -> tuple[list[str], None, list[str | None], dict]:
    import yaml

    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML object in {config_path}")
    run_dirs, store_ids = _expand_store_config(config)
    return run_dirs, None, store_ids, config


def _setting(args: argparse.Namespace, config: dict, name: str, default):
    cli_value = getattr(args, name)
    return cli_value if cli_value is not None else config.get(name, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pseudonymized action/image/user configs with base64 image "
            "payloads."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        help="YAML file defining one or more store_id/run_dir mappings.",
    )
    source.add_argument("--run_dir", dest="run_dirs", nargs="+")
    parser.add_argument(
        "--raw_data",
        dest="raw_data_dirs",
        nargs="+",
        help="One raw-data folder per --run_dir, used only to infer store_id.",
    )
    parser.add_argument(
        "--store_id",
        dest="store_ids",
        nargs="+",
        help="Optional explicit store id per --run_dir (otherwise inferred from raw_data).",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of complete runs/sessions to sample before train/test splitting.",
    )
    parser.add_argument(
        "--post_verify_threshold",
        type=float,
        default=None,
        help=(
            "Drop runs whose mean post_verify_result.json score is below this value "
            "before sampling and splitting. Runs without scores are kept."
        ),
    )
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--split_by",
        choices=["user", "session"],
        default=None,
        help="Keep users (default) or only sessions disjoint across train/test.",
    )
    parser.add_argument("--push_to_hub", action="store_true", default=None)
    parser.add_argument("--repo_id", default=None)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    yaml_config: dict = {}
    if args.config:
        if args.raw_data_dirs or args.store_ids:
            raise SystemExit("--raw_data/--store_id cannot be combined with --config")
        run_dirs, raw_data_dirs, store_ids, yaml_config = _load_yaml_config(args.config)
    else:
        run_dirs = args.run_dirs
        raw_data_dirs = args.raw_data_dirs
        store_ids = args.store_ids

    output_dir = _setting(args, yaml_config, "output_dir", "data/hf_publish_v3")
    sample = _setting(args, yaml_config, "sample", None)
    threshold = _setting(args, yaml_config, "post_verify_threshold", None)
    test_ratio = _setting(args, yaml_config, "test_ratio", 0.2)
    seed = _setting(args, yaml_config, "seed", 42)
    split_by = _setting(args, yaml_config, "split_by", "user")
    push = _setting(args, yaml_config, "push_to_hub", False)
    repo_id = _setting(args, yaml_config, "repo_id", None)
    private = _setting(args, yaml_config, "private", True)
    if push and not repo_id:
        raise SystemExit("--repo_id is required with --push_to_hub")
    built_paths, published_statistics = build_hf_dataset(
        run_dirs=run_dirs,
        raw_data_dirs=raw_data_dirs,
        store_ids=store_ids,
        output_dir=output_dir,
        test_ratio=test_ratio,
        seed=seed,
        split_by=split_by,
        sample=sample,
        post_verify_threshold=threshold,
        return_statistics=True,
    )
    for config in built_paths:
        print(f"{config}: " + ", ".join(
            f"{split}={built_paths[config][split]}" for split in CONFIG_SPLITS[config]
        ))
    if push:
        statistics_path = Path(output_dir) / "dataset_statistics.json"
        push_to_hub(
            built_paths,
            repo_id,
            private=private,
        )
        print(f"statistics={statistics_path}")
        print(json.dumps(published_statistics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
