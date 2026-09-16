"""Sample complete buyer-sim sessions and publish one or more Hub datasets.

The source must use the normalized multi-config layout produced by
``user_model.publish_data_to_hf``. Selected sessions keep their original train/test
split, and all linked configurations are filtered consistently:

* ``action`` and ``user`` are filtered by ``(store_id, session_id)``.
* ``image`` is filtered to image IDs referenced by selected actions.

Example::

    python -m user_model.sft.data_sampling \
        --output_repo <huggingface_repo> \
        --sample_size 10 \
        --clean_actions --control_add_to_cart --add_to_cart_ratio 0.10 \
        --private

Publish nested samples in one run by pairing repositories and per-store sizes::

    python -m user_model.sft.data_sampling \
        --output_repo <huggingface_repo_1> <huggingface_repo_2> \
        --sample_size 10 20 \
        --private

The default source is ``<huggingface_repo_id>/buyer-sim-complete-v4``. Its five verified
store IDs are selected by default; pass ``--store_id`` to use a subset.

The output remains normalized. It can therefore be passed directly as
``--synthetic_repo`` to both ``user_model.sft.data_prep`` and
``user_model.rl.data_prep``.

``preprocess_data`` runs before session sampling. By default it removes actions
outside ``click/type/terminate``, drops sessions shorter than ``--min_steps``,
and controls add-to-cart sessions to ``--add_to_cart_ratio``.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Iterable, Mapping
from typing import Any

from user_model.sft.data_prep import reconstruct_sessions
from user_model.prompt_templates import normalize_assistant_payload
from user_model.sft.data_loader import Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REQUIRED_CONFIGS = ("action", "user")
OPTIONAL_CONFIGS = ("image")
DEFAULT_SYNTHETIC_REPO = "<huggingface_repo_id>/buyer-sim-complete-v4"
DEFAULT_STORE_IDS = (
    "02d4a794-3b85-3bb8-c545-6711dd50a3d7",
    "2cc1259e-4ea0-c71c-ba7c-d6d4de7fb827",
    "68e1563a-676a-694b-ce31-820f79080bad",
    "8b1e812c-2f08-af73-0789-d6da90ab43ee",
    "e7d14bc5-e5b3-ffe5-47b1-9dba84f5dfe3",
)

WHITELIST_ACTIONS = {"click", "type", "terminate"}
_ADD_TARGET_SIGNALS = ("add_to_cart", "add_to_bag", "add_item")
_ADD_DESC_PHRASES = ("add to cart", "add to bag", "add item")


def _is_add_to_cart(action: dict) -> bool:
    target = str(action.get("target") or "").lower()
    description = str(action.get("description") or "").lower()
    return any(signal in target for signal in _ADD_TARGET_SIGNALS) or any(
        phrase in description for phrase in _ADD_DESC_PHRASES
    )


def _session_has_add_to_cart(session: Session) -> bool:
    return any(_is_add_to_cart(step.action) for step in session.steps)


def _clean_session(session: Session) -> None:
    """Keep only action types used by the low-resource OPeRA experiment."""
    session.steps = [
        step
        for step in session.steps
        if str(step.action.get("action") or "").lower() in WHITELIST_ACTIONS
    ]


def _truncate_before_first_add(session: Session) -> Session | None:
    first_add = next(
        (
            index
            for index, step in enumerate(session.steps)
            if _is_add_to_cart(step.action)
        ),
        None,
    )
    if first_add is None:
        return session
    if first_add == 0:
        return None
    return Session(
        session_id=session.session_id,
        persona=session.persona,
        intent=session.intent,
        steps=session.steps[:first_add],
    )


def _control_add_to_cart(
    sessions: list[Session],
    ratio: float,
    rng: random.Random,
    target_size: int | None = None,
    min_steps: int = 2,
) -> list[Session]:
    """Cap add-to-cart sessions at ``ratio`` and optionally salvage prefixes."""
    if ratio >= 1:
        return sessions

    add_sessions = [session for session in sessions if _session_has_add_to_cart(session)]
    other_sessions = [
        session for session in sessions if not _session_has_add_to_cart(session)
    ]
    keep_count = (
        int(ratio * len(other_sessions) / (1 - ratio)) if ratio > 0 else 0
    )
    if len(add_sessions) <= keep_count:
        return sessions

    shuffled = list(add_sessions)
    rng.shuffle(shuffled)
    kept = shuffled[:keep_count] + other_sessions
    salvaged = 0
    if target_size is not None and len(kept) < target_size:
        for session in shuffled[keep_count:]:
            if len(kept) >= target_size:
                break
            truncated = _truncate_before_first_add(session)
            if truncated is not None and len(truncated.steps) >= min_steps:
                kept.append(truncated)
                salvaged += 1

    logger.info(
        "Controlled add-to-cart sessions: kept %d/%d add sessions and %d "
        "non-add sessions%s",
        keep_count,
        len(add_sessions),
        len(other_sessions),
        f"; salvaged {salvaged} browse prefixes" if salvaged else "",
    )
    return kept


def _string(value: object) -> str:
    return "" if value is None else str(value)


def _session_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return _string(row.get("store_id")), _string(row.get("session_id"))


def _resolved_sample_sizes(
    store_ids: list[str], sample_sizes: list[int]
) -> list[int]:
    if not sample_sizes:
        raise ValueError("At least one --sample_size is required")
    if len(sample_sizes) == 1:
        sizes = sample_sizes * len(store_ids)
    elif len(sample_sizes) == len(store_ids):
        sizes = list(sample_sizes)
    else:
        raise ValueError(
            "--sample_size expects one value or one value per --store_id "
            f"({len(store_ids)} stores, {len(sample_sizes)} values)"
        )
    if any(size <= 0 for size in sizes):
        raise ValueError("--sample_size values must be positive")
    return sizes


def select_session_keys(
    action_splits: Mapping[str, Iterable[Mapping[str, Any]]],
    user_splits: Mapping[str, Iterable[Mapping[str, Any]]],
    store_ids: list[str],
    sample_sizes: list[int],
    seed: int = 42,
) -> tuple[set[tuple[str, str]], dict[str, dict[str, int]]]:
    """Select eligible complete-session keys independently for each store."""
    selected_stores = [str(store_id) for store_id in store_ids]
    if not selected_stores:
        raise ValueError("At least one --store_id is required")
    if len(set(selected_stores)) != len(selected_stores):
        raise ValueError("--store_id values must be unique")
    sizes = _resolved_sample_sizes(selected_stores, sample_sizes)

    action_keys = {
        key
        for rows in action_splits.values()
        for row in rows
        for key in [_session_key(row)]
        if key[0] and key[1]
    }
    user_key_splits: dict[tuple[str, str], str] = {}
    for split, rows in user_splits.items():
        for row in rows:
            key = _session_key(row)
            if not key[0] or not key[1]:
                continue
            previous = user_key_splits.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    f"Session store_id={key[0]} session_id={key[1]} appears in "
                    f"both {previous!r} and {split!r} user splits"
                )

    eligible_by_store: dict[str, list[tuple[str, str]]] = {
        store_id: sorted(
            key
            for key in user_key_splits
            if key[0] == store_id and key in action_keys
        )
        for store_id in selected_stores
    }
    missing = [store_id for store_id, keys in eligible_by_store.items() if not keys]
    if missing:
        available = sorted({key[0] for key in user_key_splits if key in action_keys})
        raise ValueError(
            f"Requested store(s) have no complete sessions: {missing}; "
            f"available stores: {available}"
        )

    rng = random.Random(seed)
    selected: set[tuple[str, str]] = set()
    stats: dict[str, dict[str, int]] = {}
    for store_id, size in zip(selected_stores, sizes):
        eligible = eligible_by_store[store_id]
        if size > len(eligible):
            raise ValueError(
                f"Requested sample_size {size} for store_id={store_id}, but only "
                f"{len(eligible)} complete session(s) are available"
            )
        # Shuffle the complete eligible pool, then take a prefix. Unlike calling
        # random.sample with different k values, this guarantees that repeated
        # requests with the same stores and seed are nested: size 10 is a strict
        # subset of size 20 (when enough sessions exist).
        ordered = list(eligible)
        rng.shuffle(ordered)
        sampled = ordered[:size]
        selected.update(sampled)
        stats[store_id] = {"available": len(eligible), "sampled": len(sampled)}
        logger.info("Store %s: %s", store_id, stats[store_id])
    return selected, stats


def _filter_session_dataset(dataset, selected_keys: set[tuple[str, str]]):
    return dataset.filter(
        lambda store_id, session_id: (
            _string(store_id),
            _string(session_id),
        )
        in selected_keys,
        input_columns=["store_id", "session_id"],
    )


def _filter_value_dataset(dataset, column: str, selected_values: set[str]):
    return dataset.filter(
        lambda value: _string(value) in selected_values,
        input_columns=[column],
    )


def _is_clean_action(action_json: object) -> bool:
    payload = normalize_assistant_payload(action_json)
    return bool(
        payload
        and _string(payload.get("action")).lower() in WHITELIST_ACTIONS
    )


def _all_rows(splits: Mapping[str, Iterable[Mapping[str, Any]]]):
    for rows in splits.values():
        yield from rows


def preprocess_data(
    action_source,
    user_source,
    store_ids: list[str],
    seed: int,
    clean_actions: bool,
    control_add_to_cart: bool,
    add_to_cart_ratio: float,
    min_steps: int,
):
    """Clean actions and control add-to-cart sessions before sampling.

    ``target_size`` is intentionally not passed to add-to-cart control. This
    makes the eligible pool independent of the later requested sample size,
    which is required for cumulative datasets to remain nested.
    """
    from datasets import DatasetDict

    if not 0 <= add_to_cart_ratio <= 1:
        raise ValueError("--add_to_cart_ratio must be between 0 and 1")
    if min_steps <= 0:
        raise ValueError("--min_steps must be positive")

    if clean_actions:
        action_source = DatasetDict(
            {
                split: rows.filter(
                    _is_clean_action,
                    input_columns=["action_json"],
                )
                for split, rows in action_source.items()
            }
        )

    sessions_by_store = reconstruct_sessions(
        _all_rows(action_source),
        _all_rows(user_source),
        min_steps=min_steps,
    )
    rng = random.Random(seed)
    controlled_keys: set[tuple[str, str]] = set()
    preprocessing_stats: dict[str, dict[str, int]] = {}
    for store_id in [str(value) for value in store_ids]:
        sessions = list(sessions_by_store.get(store_id, []))
        if clean_actions:
            # The normalized rows were already filtered so published actions
            # match the cleaned Session objects. Reusing the shared helper here
            # keeps the action-space contract identical to data_prep.py.
            for session in sessions:
                _clean_session(session)
            sessions = [session for session in sessions if len(session.steps) >= min_steps]
        after_cleaning = len(sessions)
        if control_add_to_cart:
            sessions = _control_add_to_cart(
                sessions,
                add_to_cart_ratio,
                rng,
                target_size=None,
                min_steps=min_steps,
            )
        after_control = len(sessions)
        controlled_keys.update(
            (store_id, session.session_id) for session in sessions
        )
        preprocessing_stats[store_id] = {
            "after_cleaning": after_cleaning,
            "after_add_to_cart_control": after_control,
        }
        logger.info("Preprocessed store %s: %s", store_id, preprocessing_stats[store_id])

    eligible_actions = DatasetDict(
        {
            split: _filter_session_dataset(rows, controlled_keys)
            for split, rows in action_source.items()
        }
    )
    eligible_users = DatasetDict(
        {
            split: _filter_session_dataset(rows, controlled_keys)
            for split, rows in user_source.items()
        }
    )
    return eligible_actions, eligible_users, preprocessing_stats


def build_sampled_dataset(
    synthetic_repo: str,
    store_ids: list[str],
    sample_sizes: list[int],
    seed: int = 42,
    clean_actions: bool = True,
    control_add_to_cart: bool = True,
    add_to_cart_ratio: float = 0.10,
    min_steps: int = 2,
    revision: str | None = None,
    token: str | None = None,
):
    """Load and consistently filter all normalized source configurations."""
    from datasets import DatasetDict, get_dataset_config_names, load_dataset

    config_names = set(
        get_dataset_config_names(synthetic_repo, revision=revision, token=token)
    )
    missing = [config for config in REQUIRED_CONFIGS if config not in config_names]
    if missing:
        raise ValueError(
            f"Source dataset {synthetic_repo!r} is missing required config(s): {missing}"
        )

    logger.info("Loading action and user configs from %s", synthetic_repo)
    action_source = load_dataset(
        synthetic_repo, "action", revision=revision, token=token
    )
    user_source = load_dataset(
        synthetic_repo, "user", revision=revision, token=token
    )
    eligible_actions, eligible_users, preprocessing_stats = preprocess_data(
        action_source=action_source,
        user_source=user_source,
        store_ids=store_ids,
        seed=seed,
        clean_actions=clean_actions,
        control_add_to_cart=control_add_to_cart,
        add_to_cart_ratio=add_to_cart_ratio,
        min_steps=min_steps,
    )
    selected_keys, stats = select_session_keys(
        eligible_actions,
        eligible_users,
        store_ids=store_ids,
        sample_sizes=sample_sizes,
        seed=seed,
    )

    sampled: dict[str, Any] = {
        "action": DatasetDict(
            {
                split: _filter_session_dataset(rows, selected_keys)
                for split, rows in eligible_actions.items()
            }
        ),
        "user": DatasetDict(
            {
                split: _filter_session_dataset(rows, selected_keys)
                for split, rows in eligible_users.items()
            }
        ),
    }

    if "image" in config_names:
        image_source = load_dataset(
            synthetic_repo, "image", revision=revision, token=token
        )
        image_ids_by_split = {
            split: {
                _string(row.get("image"))
                for row in rows
                if row.get("image") not in (None, "")
            }
            for split, rows in sampled["action"].items()
        }
        sampled["image"] = DatasetDict(
            {
                split: _filter_value_dataset(
                    rows,
                    "image_id",
                    image_ids_by_split.get(split, set()),
                )
                for split, rows in image_source.items()
            }
        )

    counts = {
        config: {split: len(rows) for split, rows in splits.items()}
        for config, splits in sampled.items()
    }
    logger.info("Sampled normalized dataset rows: %s", counts)
    return sampled, {
        "preprocessing": {
            "clean_actions": clean_actions,
            "control_add_to_cart": control_add_to_cart,
            "add_to_cart_ratio": add_to_cart_ratio,
            "min_steps": min_steps,
            "stores": preprocessing_stats,
        },
        "stores": stats,
        "rows": counts,
    }


def sample_and_push_dataset(
    synthetic_repo: str,
    output_repo: str,
    store_ids: list[str],
    sample_sizes: list[int],
    seed: int = 42,
    clean_actions: bool = True,
    control_add_to_cart: bool = True,
    add_to_cart_ratio: float = 0.10,
    min_steps: int = 2,
    private: bool = True,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Build a normalized sample and push each configuration to one Hub repo."""
    if synthetic_repo == output_repo:
        raise ValueError("--output_repo must differ from --synthetic_repo")
    sampled, stats = build_sampled_dataset(
        synthetic_repo=synthetic_repo,
        store_ids=store_ids,
        sample_sizes=sample_sizes,
        seed=seed,
        clean_actions=clean_actions,
        control_add_to_cart=control_add_to_cart,
        add_to_cart_ratio=add_to_cart_ratio,
        min_steps=min_steps,
        revision=revision,
        token=token,
    )
    for config, dataset in sampled.items():
        logger.info("Pushing %s config to %s", config, output_repo)
        dataset.push_to_hub(
            output_repo,
            config_name=config,
            private=private,
            token=token,
            commit_message=(
                f"Sample {sum(item['sampled'] for item in stats['stores'].values())} "
                f"synthetic sessions for {config}"
            ),
        )
    logger.info("Published sampled dataset at https://huggingface.co/datasets/%s", output_repo)
    return stats


def _output_plan(
    synthetic_repo: str,
    output_repos: list[str],
    store_ids: list[str],
    sample_sizes: list[int],
) -> list[tuple[str, list[int]]]:
    """Resolve CLI values into ``(output_repo, per-store sizes)`` jobs.

    One output repository preserves the original behavior: one sample size is
    broadcast to all stores, or one size may be supplied per store. Multiple
    output repositories use cumulative-tier behavior: each repository is
    paired with one strictly increasing sample size that is broadcast to every
    selected store.
    """
    if not output_repos:
        raise ValueError("At least one --output_repo is required")
    if len(set(output_repos)) != len(output_repos):
        raise ValueError("--output_repo values must be unique")
    if synthetic_repo in output_repos:
        raise ValueError("Every --output_repo must differ from --synthetic_repo")
    if not store_ids:
        raise ValueError("At least one --store_id is required")

    if len(output_repos) == 1:
        _resolved_sample_sizes(store_ids, sample_sizes)
        return [(output_repos[0], list(sample_sizes))]

    if len(output_repos) != len(sample_sizes):
        raise ValueError(
            "Multiple --output_repo values require one matching --sample_size "
            f"each ({len(output_repos)} repositories, {len(sample_sizes)} sizes)"
        )
    if any(size <= 0 for size in sample_sizes):
        raise ValueError("--sample_size values must be positive")
    if any(later <= earlier for earlier, later in zip(sample_sizes, sample_sizes[1:])):
        raise ValueError(
            "With multiple outputs, --sample_size values must be strictly increasing"
        )
    return [
        (output_repo, [sample_size])
        for output_repo, sample_size in zip(output_repos, sample_sizes)
    ]


def sample_and_push_datasets(
    synthetic_repo: str,
    output_repos: list[str],
    store_ids: list[str],
    sample_sizes: list[int],
    seed: int = 42,
    clean_actions: bool = True,
    control_add_to_cart: bool = True,
    add_to_cart_ratio: float = 0.10,
    min_steps: int = 2,
    private: bool = True,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Publish one dataset or a cumulative series from one sampling command."""
    plan = _output_plan(
        synthetic_repo, output_repos, store_ids, sample_sizes
    )
    results: dict[str, dict[str, Any]] = {}
    for output_repo, per_store_sizes in plan:
        logger.info(
            "Publishing sample sizes %s to %s",
            per_store_sizes,
            output_repo,
        )
        results[output_repo] = sample_and_push_dataset(
            synthetic_repo=synthetic_repo,
            output_repo=output_repo,
            store_ids=store_ids,
            sample_sizes=per_store_sizes,
            seed=seed,
            clean_actions=clean_actions,
            control_add_to_cart=control_add_to_cart,
            add_to_cart_ratio=add_to_cart_ratio,
            min_steps=min_steps,
            private=private,
            revision=revision,
            token=token,
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample normalized synthetic sessions and push one or more nested "
            "datasets to the Hub."
        )
    )
    parser.add_argument(
        "--synthetic-repo",
        "--synthetic_repo",
        default=DEFAULT_SYNTHETIC_REPO,
        help=f"Normalized source dataset (default: {DEFAULT_SYNTHETIC_REPO}).",
    )
    parser.add_argument(
        "--output-repo",
        "--output_repo",
        dest="output_repos",
        nargs="+",
        required=True,
        help=(
            "One destination repository, or multiple repositories paired with "
            "--sample_size tiers."
        ),
    )
    parser.add_argument(
        "--store-id",
        "--store_id",
        dest="store_ids",
        nargs="+",
        default=list(DEFAULT_STORE_IDS),
        metavar="STORE_ID",
        help=(
            "Store UUIDs to sample. By default all five stores verified in "
            f"{DEFAULT_SYNTHETIC_REPO} are used: {', '.join(DEFAULT_STORE_IDS)}"
        ),
    )
    parser.add_argument(
        "--sample-size",
        "--sample_size",
        dest="sample_sizes",
        nargs="+",
        required=True,
        type=int,
        help=(
            "For one output: one value for all stores or one per --store_id. "
            "For multiple outputs: one increasing value per --output_repo."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean_actions",
        "--cleaning",
        dest="clean_actions",
        action="store_true",
        default=True,
        help=f"Keep only action types {sorted(WHITELIST_ACTIONS)} before sampling (default).",
    )
    parser.add_argument(
        "--no_clean_actions",
        "--no_cleaning",
        dest="clean_actions",
        action="store_false",
    )
    cart_control = parser.add_mutually_exclusive_group()
    cart_control.add_argument(
        "--control_add_to_cart",
        dest="control_add_to_cart",
        action="store_true",
        help="Control the add-to-cart session ratio before sampling (default).",
    )
    cart_control.add_argument(
        "--no_add_to_cart_sampling",
        dest="control_add_to_cart",
        action="store_false",
        help="Disable add-to-cart ratio control before sampling.",
    )
    parser.add_argument("--add_to_cart_ratio", type=float, default=0.10)
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--revision")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True, control_add_to_cart=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sample_and_push_datasets(
        synthetic_repo=args.synthetic_repo,
        output_repos=args.output_repos,
        store_ids=args.store_ids,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
        clean_actions=args.clean_actions,
        control_add_to_cart=args.control_add_to_cart,
        add_to_cart_ratio=args.add_to_cart_ratio,
        min_steps=args.min_steps,
        private=args.private,
        revision=args.revision,
    )
