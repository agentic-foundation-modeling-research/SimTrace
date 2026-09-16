"""Shared skill-probe helpers for build-loop verifiers (impl plan T1.0).

The :class:`~shop_arena.gen.build.verifiers.visual_judge.VisualJudgeVerifier`
(M1) and the final-eval visual sweep (M5) both depend on the
``pi-playwright`` skill being installed. The probe checks the workspace
``node_modules`` install first, then global JS package-manager roots.
When the skill is missing the visual judge is omitted from the verifier
tuple with a single warning, per spec §5.5.1.

The canonical resolver lives in :mod:`shop_arena.explore.pipeline` (where it
also gates the playwright session pre-open). This module imports it
and adds the boolean predicate that the verifier factory needs.

This module is import-safe: no I/O, no env reads, and no side effects
at import time. Callers invoke :func:`is_playwright_skill_available`
(which shells out to ``pnpm root -g`` / ``npm root -g`` via the lifted
resolver) only when they need to decide whether to register the
visual judge.
"""

from __future__ import annotations

from pathlib import Path

import playwright.sync_api

from shop_arena.explore.pipeline import resolve_playwright_skill_dir

__all__ = [
    "is_playwright_skill_available",
    "is_python_playwright_available",
]


def is_playwright_skill_available() -> bool:
    """Return ``True`` iff the playwright skill is fully usable.

    "Fully usable" means the skill directory resolves *and* its
    ``scripts/pw.js`` entrypoint is a regular file. The latter check
    catches partial installs where the package directory exists but
    the CLI shim was not laid down. Spec §5.5.1.

    Returns:
        ``True`` when the resolver finds the skill and ``pw.js`` is
        present; ``False`` otherwise.
    """
    skill_dir = resolve_playwright_skill_dir()
    if skill_dir is None:
        return False
    return (skill_dir / "scripts" / "pw.js").is_file()


def is_python_playwright_available() -> bool:
    """Return ``True`` iff the Python ``playwright`` package + Chromium are usable.

    Used to gate the
    :class:`~shop_arena.gen.build.verifiers.clickstream_replay.ClickstreamReplayVerifier`,
    which drives ``playwright.sync_api`` directly (not the ``pi-playwright``
    JS skill). The ``playwright`` distribution is a hard dependency, so
    the only thing in question is whether its bundled Chromium is
    installed on disk (``executable_path`` resolves to an existing file).

    The check starts and immediately stops the Playwright driver; it
    never launches a browser. Any failure (driver error, absent browser
    binary) resolves to ``False`` so the verifier factory can
    warn-and-omit rather than crash the build loop.

    Returns:
        ``True`` when Chromium is installed and launchable; ``False``
        otherwise.
    """
    try:
        with playwright.sync_api.sync_playwright() as pw:
            return Path(pw.chromium.executable_path).is_file()
    except Exception:  # noqa: BLE001 -- any probe failure means "unavailable"
        return False
