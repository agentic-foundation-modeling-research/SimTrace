"""Generate buyer personas for sessions.jsonl from precomputed buyer features.

Reads the per-buyer feature dict produced by
``bigquery_processor.py`` (``buyers_feature.json``, keyed by
buyer_id) and a ``sessions.jsonl`` whose records carry a ``persona_id`` (= the
buyer_id). For the first ``--n`` buyers appearing in ``sessions.jsonl``, it calls
the persona LLM (``persona_prompt.persona_sys_prompt``) once per buyer and writes
the raw JSON profile into every matching record's ``persona`` field.

Idempotent: buyers whose records already have a non-empty ``persona`` are skipped
unless ``--overwrite`` is given. The original sessions.jsonl is backed up once to
``sessions.jsonl.pre_persona``.

Usage:
    python -m src.data_collection.generate_personas \
        --features <path/to/buyers_feature.json> \
        --sessions <path/to/sessions.jsonl> \
        -n 20
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# Sibling import (persona_prompt) + repo-root import (src.*), regardless of how
# the script is invoked.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]  # src/data_collection/ -> repository root
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

try:
    from .persona_prompt import persona_sys_prompt
except ImportError:  # direct-file execution fallback
    from persona_prompt import persona_sys_prompt
try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Defaults mirror conf/base.yaml's configured LLM proxy.
DEFAULT_MODEL = "googlevertexai-global:gemini-3-flash-preview"
DEFAULT_PROVIDER = "openai"
DEFAULT_BASE_URL = os.getenv("DEFAULT_BASE_URL")
LITELLM_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write to a sibling temp file then atomically replace, so an interrupted
    # rewrite can never leave a truncated file on disk.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _buyer_order(records: list[dict]) -> list[str]:
    """Distinct persona_ids in first-appearance order (empty ids ignored)."""
    seen: dict[str, None] = {}
    for r in records:
        pid = str(r.get("persona_id") or "")
        if pid and pid not in seen:
            seen[pid] = None
    return list(seen)


def _already_done(records: list[dict]) -> set[str]:
    """Buyer ids that already have a non-empty persona on at least one record."""
    done = set()
    for r in records:
        if str(r.get("persona") or "").strip():
            pid = str(r.get("persona_id") or "")
            if pid:
                done.add(pid)
    return done


async def _gen_one(
    buyer_id: str,
    features: dict,
    llm_config: SimpleNamespace,
    sem: asyncio.Semaphore,
) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": persona_sys_prompt},
        {"role": "user", "content": json.dumps(features, ensure_ascii=False)},
    ]
    async with sem:
        try:
            # Delay the simulation stack import until generation actually starts
            # so documentation and ``--help`` do not require model dependencies.
            from src.data_gen.agent.gpt import async_chat

            persona = await async_chat(
                messages,
                model_name=llm_config.model,
                provider=llm_config.provider,
                llm_config=llm_config,
                json_mode=True,
            )
            return buyer_id, persona.strip()
        except Exception as exc:  # terminal failure after retries
            logger.warning("Persona generation failed for %s: %s", buyer_id, exc)
            return buyer_id, ""


async def _shutdown_litellm(
    timeout: float = LITELLM_SHUTDOWN_TIMEOUT_SECONDS,
) -> None:
    """Tear down litellm's lingering event-loop resources.

    litellm starts a background logging worker and keeps async HTTP client
    sessions open on the running loop. Left behind, they stall asyncio.run()'s
    shutdown (it blocks in _cancel_all_tasks waiting on the worker), so the
    process appears to hang after all work is done. Stop them explicitly, but
    never let best-effort cleanup hold a completed pipeline open indefinitely.
    """
    try:
        import litellm
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except ImportError:
        return

    async def _close_resources() -> None:
        with contextlib.suppress(Exception):
            await GLOBAL_LOGGING_WORKER.stop()
        with contextlib.suppress(Exception):
            await litellm.close_litellm_async_clients()

    try:
        await asyncio.wait_for(_close_resources(), timeout=timeout)
    except TimeoutError:
        logger.warning(
            "LiteLLM cleanup exceeded %.1f seconds; continuing shutdown",
            timeout,
        )


async def run(args: argparse.Namespace) -> None:
    features_path = Path(args.features)
    sessions_path = Path(args.sessions)
    features: dict[str, dict] = json.loads(features_path.read_text(encoding="utf-8"))
    records = _read_jsonl(sessions_path)

    order = _buyer_order(records)
    target = order[: args.n]
    done = set() if args.overwrite else _already_done(records)

    to_generate = []
    for buyer_id in target:
        if buyer_id in done:
            continue
        if buyer_id not in features:
            logger.warning("Buyer %s not found in %s; skipping", buyer_id, features_path)
            continue
        to_generate.append(buyer_id)

    logger.info(
        "Buyers in sessions: %d; targeting first %d; generating %d (skipped %d done)",
        len(order), len(target), len(to_generate), len(target) - len(to_generate),
    )
    if not to_generate:
        logger.info("Nothing to generate.")
        return

    llm_config = SimpleNamespace(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        enable_thinking=None,
        request_timeout=args.request_timeout,
    )
    # persona_id -> record indices, so each completed persona updates its
    # records in O(1) without rescanning the whole list.
    pid_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        pid = str(r.get("persona_id") or "")
        if pid:
            pid_to_indices.setdefault(pid, []).append(i)

    # Snapshot the original before we start rewriting it in place.
    backup = sessions_path.with_suffix(sessions_path.suffix + ".pre_persona")
    if not backup.exists():
        backup.write_text(sessions_path.read_text(encoding="utf-8"), encoding="utf-8")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(_gen_one(b, features[b], llm_config, sem))
        for b in to_generate
    ]
    bar = (
        tqdm(total=len(tasks), desc="Generating personas", unit="buyer")
        if tqdm
        else None
    )
    generated = 0
    updated = 0
    # Persist each persona as soon as it comes back, so an interruption only
    # costs the in-flight calls rather than the whole run.
    try:
        for fut in asyncio.as_completed(tasks):
            buyer_id, persona = await fut
            if bar is not None:
                bar.update(1)
            if not persona:
                continue
            for i in pid_to_indices.get(buyer_id, []):
                records[i]["persona"] = persona
                updated += 1
            generated += 1
            _write_jsonl(sessions_path, records)  # crash-safe checkpoint
    finally:
        if bar is not None:
            bar.close()
        # Release litellm's background worker / async clients so the process
        # exits cleanly instead of hanging in asyncio shutdown.
        await _shutdown_litellm()

    logger.info(
        "Generated %d personas; updated %d records in %s",
        generated, updated, sessions_path,
    )


def generate_personas(
    features: str | Path,
    sessions: str | Path,
    n: int | None = None,
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    base_url: str = DEFAULT_BASE_URL,
    concurrency: int = 16,
    request_timeout: float = 600,
    overwrite: bool = False,
) -> None:
    """Generate personas as a callable enrichment step.

    ``n=None`` targets every buyer referenced by ``sessions``. The standalone
    CLI and :mod:`enrich_sessions` both use this function so persona generation
    has one implementation.
    """
    sessions_path = Path(sessions)
    if n is None:
        n = len(_buyer_order(_read_jsonl(sessions_path)))
    args = argparse.Namespace(
        features=str(features),
        sessions=str(sessions_path),
        n=n,
        model=model,
        provider=provider,
        base_url=base_url,
        concurrency=concurrency,
        request_timeout=request_timeout,
        overwrite=overwrite,
    )
    asyncio.run(run(args))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="buyers_feature.json path")
    parser.add_argument("--sessions", required=True, help="sessions.jsonl (edited in place)")
    parser.add_argument(
        "-n", type=int, required=True,
        help="Generate personas for the first N buyers appearing in sessions.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate even for buyers whose persona is already set",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    generate_personas(
        features=cli_args.features,
        sessions=cli_args.sessions,
        n=cli_args.n,
        model=cli_args.model,
        provider=cli_args.provider,
        base_url=cli_args.base_url,
        concurrency=cli_args.concurrency,
        request_timeout=cli_args.request_timeout,
        overwrite=cli_args.overwrite,
    )
