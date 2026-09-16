"""Build the final method-comparison table and paired-bootstrap delta CIs.

Part 1 — aggregate every method's pooled ``eval_results_simplified.json`` into
``final_table.csv`` (first column ``Method``).

Part 2 — paired bootstrap (shared index set per iteration) of the delta between
``Ours`` and every baseline for each headline metric. The headline metrics are
computed over pooled session distributions (see ``aggregate_runs``), so they are
recomputed on every resample from per-session statistics precomputed once:

    for b in 1..B:
        idx = sample(1..n, size=n, replace=True)   # ONE shared index set
        delta_b = M(real[idx], ours[idx]) - M(real[idx], baseline[idx])
    CI = percentiles(delta_b, [2.5, 97.5]); significant if CI excludes 0

Part 3 — paired permutation test (Fisher randomization test). Both methods'
synthetic sessions are paired to the same real sessions, so under the null
"Ours and baseline are exchangeable" we randomly swap, per session, which
method's synth session carries the Ours label and recompute the delta:

    for p in 1..P:
        mask = coin flips over sessions       # ONE shared mask per iteration
        A, B = label-swapped(ours, baseline, mask)
        delta_p = M(real, A) - M(real, B)
    p_two = (1 + #{|delta_p| >= |delta_obs|}) / (P + 1)

One-sided p-values test the direction "Ours is better": lower for the
divergence/distance metrics, higher for ``product_sim_per_session``, and
closer to 0 (lower absolute value) for ``product_diversity_ratio``.

``rationale_similarity`` is reported in the main table but excluded from the
bootstrap and permutation test: only its mean/std are persisted, and
regenerating per-pair values would require fresh (stochastic) LLM calls.

Part 4 — post-verify (on by default, ``--no-post-verify-ours`` to disable). The
standalone post verifier (``scripts.post_verify``) is run over every pooled
source run of ``Ours``; sessions whose mean verifier score falls below
``--pv-threshold`` are excluded (same rule as ``eval.run_eval``). The main table
gains an ``Ours_postverify`` row with the metrics recomputed on the surviving
sessions (``rationale_similarity`` left blank — per-pair values are not
persisted), and ``bootstrap_ci_post_verify.csv`` repeats the paired
bootstrap/permutation comparison of filtered-Ours vs every baseline restricted
to the *same* surviving session ids, so both sides of each comparison use one
population. ``bootstrap_ci.csv`` remains the unfiltered comparison.

Usage:
    python -m eval.final_table --results-dir results_toy_table_test \
        --output-dir results_toy_table_test --bootstrap 10000 \
        --permutations 10000 --seed 42
"""

import argparse
import asyncio
import csv
import json
import logging
import re
from dataclasses import fields
from pathlib import Path

import numpy as np

from eval.models import NormalizedSession, SEMANTIC_ACTIONS
from eval.outcome_level_fidelity import OutcomeLevelFidelity
from eval.run_eval import (
    _compute_post_verify_exclusions,
    _DEFAULT_POST_VERIFY_THRESHOLD,
)
from eval.semantic_level_fidelity import SemanticLevelFidelity
from eval.sequence_level_fidelity import SequenceLevelFidelity

logger = logging.getLogger(__name__)

# Method name -> the traj_cond slug between the model prefix and the trailing
# __n<k>. ``persona`` is resolved by its first-level folder name instead.
TRAJ_COND_SLUGS = {
    "actsingle_fee0_per0_pla0_verno": "trajectory",
    "actsingle_fee0_per1_pla0_verno": "trajectory+persona",
    "actsingle_fee1_per1_pla1_verno": "Plan_Act_Reflect",
    "actsingle_fee1_per1_pla1_verpre": "Ours",
}
METHOD_ORDER = ["persona", "trajectory", "trajectory+persona", "Plan_Act_Reflect", "Ours"]
OURS = "Ours"
OURS_PV = "Ours_postverify"

METRIC_KEYS = [
    "outcome_jsd",
    "action_freq_jsd",
    "session_length_jsd",
    "transition_matrix_L1",
    "trajectory_lev_distance",
    "product_coherence_gap",
    "product_diversity_ratio",
    "product_sim_per_session",
    "rationale_similarity",
]
BOOTSTRAP_METRICS = [k for k in METRIC_KEYS if k != "rationale_similarity"]

# Direction of the one-sided hypothesis "Ours is better". product_diversity_ratio
# is best at 0, so its one-sided test is lower-is-better on |value|.
LOWER_IS_BETTER = {
    "outcome_jsd",
    "action_freq_jsd",
    "session_length_jsd",
    "transition_matrix_L1",
    "trajectory_lev_distance",
    "product_coherence_gap",
}
HIGHER_IS_BETTER = {"product_sim_per_session"}
ABS_LOWER_IS_BETTER = {"product_diversity_ratio"}

_NORMALIZED_FIELDS = {f.name for f in fields(NormalizedSession)}
_N_ACTIONS = len(SEMANTIC_ACTIONS)
_ACTION_INDEX = {a: i for i, a in enumerate(SEMANTIC_ACTIONS)}


def discover_methods(results_dir: Path) -> dict[str, Path]:
    """Map method names to experiment folders under ``results_dir``."""
    methods: dict[str, Path] = {}

    persona_root = results_dir / "persona"
    if persona_root.is_dir():
        candidates = [p for p in sorted(persona_root.iterdir()) if p.is_dir()]
        if len(candidates) > 1:
            logger.warning(
                "Multiple persona folders found, using %s", candidates[0].name
            )
        if candidates:
            methods["persona"] = candidates[0]

    traj_root = results_dir / "traj_cond"
    if traj_root.is_dir():
        for p in sorted(traj_root.iterdir()):
            if not p.is_dir() or "__" not in p.name:
                continue
            # <model>__<slug>__n<k>  ->  <slug>
            middle = re.sub(r"__n\d+$", "", p.name.split("__", 1)[1])
            method = TRAJ_COND_SLUGS.get(middle)
            if method is None:
                logger.warning("Unrecognized traj_cond slug %r (%s), skipping", middle, p.name)
            elif method in methods:
                logger.warning("Duplicate folder for method %r: %s (keeping %s)", method, p.name, methods[method].name)
            else:
                methods[method] = p
    return methods


def load_simplified(methods: dict[str, Path]) -> dict[str, dict]:
    tables = {}
    for method, folder in methods.items():
        path = folder / "eval_results_simplified.json"
        if not path.is_file():
            logger.warning("Missing %s, skipping method %r", path, method)
            continue
        with open(path) as f:
            tables[method] = json.load(f)
    return tables


def _pooled_sources(folder: Path) -> list[tuple[str, Path]]:
    """(tag, run_folder) for every run pooled into ``folder/eval_results.json``.

    ``aggregate_runs`` records each source's path and session-id tag in
    ``metadata.sources``; the tag is what prefixes the pooled session ids."""
    path = folder / "eval_results.json"
    with open(path) as f:
        sources = json.load(f).get("metadata", {}).get("sources")
    if not sources:
        raise SystemExit(
            f"{path} has no metadata.sources (pre-dates run tagging); "
            "re-run eval.aggregate_runs to regenerate it"
        )
    return [(s["tag"], Path(s["path"]).parent) for s in sources]


def post_verify_ours(
    ours_folder: Path,
    config: str,
    model: str | None,
    threshold: float,
    force: bool,
    concurrency: int,
) -> set[str]:
    """Run the standalone post verifier over every pooled source run of Ours and
    return the pooled (tag-prefixed) session ids scoring below ``threshold``.

    Runs that already hold a ``post_verify_result.json`` are skipped by the
    verifier unless ``force``, so this is idempotent and only pays LLM calls
    for unverified runs."""
    # Imported lazily: pulls in the agent stack, which the tabulation-only
    # path (--no-post-verify-ours) must not depend on.
    from scripts.post_verify import run as run_post_verifier

    excluded: set[str] = set()
    for tag, run_folder in _pooled_sources(ours_folder):
        asyncio.run(run_post_verifier(run_folder, config, model, force, concurrency))
        ids, _ = _compute_post_verify_exclusions(
            str(run_folder / "eval_results.json"), threshold
        )
        logger.info(
            "Post-verify %s: %d session(s) below threshold %.2f",
            run_folder,
            len(ids),
            threshold,
        )
        excluded.update(f"{tag}__{sid}" for sid in ids)
    return excluded


def write_main_table(
    tables: dict[str, dict],
    output: Path,
    shared_n: int | None = None,
    pv_row: tuple[dict, int, int] | None = None,
) -> None:
    """``pv_row`` is ``(metrics, n_surviving, n_excluded)`` for ``Ours_postverify``,
    inserted right after the ``Ours`` row."""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + METRIC_KEYS + ["n_sessions", "pv_num_excluded"])
        for method in METHOD_ORDER:
            if method not in tables:
                continue
            writer.writerow(
                [method]
                + [tables[method].get(k) for k in METRIC_KEYS]
                + [shared_n if shared_n is not None else "", ""]
            )
            if method == OURS and pv_row is not None:
                metrics, n_surviving, n_excluded = pv_row
                writer.writerow(
                    [OURS_PV]
                    + [metrics.get(k) for k in METRIC_KEYS]
                    + [n_surviving, n_excluded]
                )
    logger.info("Main table written to %s", output)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _load_sessions(records: list[dict]) -> dict[str, NormalizedSession]:
    """Rebuild NormalizedSession objects keyed by session_id (ids already tagged)."""
    sessions = {}
    for rec in records:
        kwargs = {k: v for k, v in rec.items() if k in _NORMALIZED_FIELDS}
        s = NormalizedSession(**kwargs)
        sessions[s.session_id] = s
    return sessions


def _outcome_flags(s: NormalizedSession) -> list[int]:
    flags = OutcomeLevelFidelity._classify_session(s)
    return [int(flags[k]) for k in OutcomeLevelFidelity.OUTCOME_LABELS]


def _action_counts(s: NormalizedSession) -> np.ndarray:
    counts = np.zeros(_N_ACTIONS)
    for a in s.actions:
        i = _ACTION_INDEX.get(a)
        if i is not None:
            counts[i] += 1
    return counts


def _transition_counts(s: NormalizedSession) -> np.ndarray:
    counts = np.zeros((_N_ACTIONS, _N_ACTIONS))
    for a_from, a_to in zip(s.actions[:-1], s.actions[1:]):
        i = _ACTION_INDEX.get(a_from)
        j = _ACTION_INDEX.get(a_to)
        if i is not None and j is not None:
            counts[i, j] += 1
    return counts


class SessionStats:
    """Per-session statistics for one side (real or synthetic), in a fixed
    session-id order, precomputed so each bootstrap resample is pure numpy."""

    def __init__(
        self,
        sessions: list[NormalizedSession],
        coherence: dict[str, float],
        pair_lev: np.ndarray | None = None,
        pair_prod_sim: np.ndarray | None = None,
    ):
        self.outcome_flags = np.array([_outcome_flags(s) for s in sessions], dtype=float)
        self.action_counts = np.array([_action_counts(s) for s in sessions])
        self.lengths = np.array([s.length for s in sessions], dtype=int)
        self.transitions = np.array([_transition_counts(s) for s in sessions])
        self.coherence = np.array(
            [coherence.get(s.session_id, np.nan) for s in sessions], dtype=float
        )
        # Paired (real vs synth by session_id) metrics; only set on synth sides.
        self.pair_lev = pair_lev
        self.pair_prod_sim = pair_prod_sim

    _ARRAY_FIELDS = (
        "outcome_flags",
        "action_counts",
        "lengths",
        "transitions",
        "coherence",
        "pair_lev",
        "pair_prod_sim",
    )

    @classmethod
    def from_arrays(cls, **arrays) -> "SessionStats":
        obj = cls.__new__(cls)
        for field in cls._ARRAY_FIELDS:
            setattr(obj, field, arrays[field])
        return obj


def _swap_stats(
    ours: SessionStats, base: SessionStats, mask: np.ndarray
) -> tuple[SessionStats, SessionStats]:
    """Label-swap two synth-side stat sets: sessions where ``mask`` is True get
    the baseline's values in the Ours slot and vice versa."""
    a, b = {}, {}
    for field in SessionStats._ARRAY_FIELDS:
        x = getattr(ours, field)
        y = getattr(base, field)
        m = mask.reshape((-1,) + (1,) * (x.ndim - 1))
        a[field] = np.where(m, y, x)
        b[field] = np.where(m, x, y)
    return SessionStats.from_arrays(**a), SessionStats.from_arrays(**b)


def _subset_stats(stats: SessionStats, keep: np.ndarray) -> SessionStats:
    """Restrict a stat set to the session positions in ``keep`` (all arrays are
    aligned on the session axis; pair_* fields are None on the real side)."""
    arrays = {}
    for field in SessionStats._ARRAY_FIELDS:
        x = getattr(stats, field)
        arrays[field] = None if x is None else x[keep]
    return SessionStats.from_arrays(**arrays)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return SequenceLevelFidelity._js_divergence_vectors(p, q)


def _length_jsd(real_lengths: np.ndarray, synth_lengths: np.ndarray) -> float:
    max_len = int(max(real_lengths.max(), synth_lengths.max())) + 1
    real_hist = np.bincount(real_lengths, minlength=max_len).astype(float)
    synth_hist = np.bincount(synth_lengths, minlength=max_len).astype(float)
    return _js_divergence(real_hist / real_hist.sum(), synth_hist / synth_hist.sum())


def _row_normalize(counts: np.ndarray) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return counts / row_sums


def compute_metrics(real: SessionStats, synth: SessionStats, idx: np.ndarray) -> dict[str, float]:
    """Recompute the 8 bootstrap metrics on one shared resample ``idx``."""
    out: dict[str, float] = {}

    real_rates = real.outcome_flags[idx].mean(axis=0)
    synth_rates = synth.outcome_flags[idx].mean(axis=0)
    out["outcome_jsd"] = _js_divergence(real_rates, synth_rates)

    real_freq = real.action_counts[idx].sum(axis=0)
    synth_freq = synth.action_counts[idx].sum(axis=0)
    out["action_freq_jsd"] = _js_divergence(
        real_freq / max(real_freq.sum(), 1), synth_freq / max(synth_freq.sum(), 1)
    )

    out["session_length_jsd"] = _length_jsd(real.lengths[idx], synth.lengths[idx])

    real_trans = _row_normalize(real.transitions[idx].sum(axis=0))
    synth_trans = _row_normalize(synth.transitions[idx].sum(axis=0))
    out["transition_matrix_L1"] = float(np.sum(np.abs(real_trans - synth_trans)))

    out["trajectory_lev_distance"] = float(np.mean(synth.pair_lev[idx]))

    real_coh = real.coherence[idx]
    synth_coh = synth.coherence[idx]
    real_mean = float(np.nanmean(real_coh)) if np.any(~np.isnan(real_coh)) else 0.0
    synth_mean = float(np.nanmean(synth_coh)) if np.any(~np.isnan(synth_coh)) else 0.0
    out["product_coherence_gap"] = abs(real_mean - synth_mean)

    n_real = int(np.sum(~np.isnan(real_coh)))
    n_synth = int(np.sum(~np.isnan(synth_coh)))
    out["product_diversity_ratio"] = n_synth / n_real - 1.0 if n_real else 0.0

    sims = synth.pair_prod_sim[idx]
    out["product_sim_per_session"] = (
        float(np.nanmean(sims)) if np.any(~np.isnan(sims)) else 0.0
    )

    return out


def load_bootstrap_inputs(
    methods: dict[str, Path],
) -> tuple[list[str], SessionStats, dict[str, SessionStats]]:
    """Load pooled eval_results.json for every method, align on the shared real
    session ids, and precompute per-session statistics."""
    raw: dict[str, dict] = {}
    for method, folder in methods.items():
        path = folder / "eval_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path} needed for bootstrap")
        with open(path) as f:
            raw[method] = json.load(f)

    real_by_method = {m: _load_sessions(d["real_sessions"]) for m, d in raw.items()}
    synth_by_method = {m: _load_sessions(d["synth_sessions"]) for m, d in raw.items()}

    shared_ids = set.intersection(*(set(r) for r in real_by_method.values()))
    shared_ids &= set.intersection(*(set(s) for s in synth_by_method.values()))
    all_ids = set.union(*(set(r) for r in real_by_method.values()))
    dropped = sorted(all_ids - shared_ids)
    if dropped:
        logger.warning(
            "Dropping %d session id(s) not present in every method: %s",
            len(dropped),
            dropped,
        )
    if not shared_ids:
        raise ValueError("No session ids shared across all methods")
    ids = sorted(shared_ids)
    logger.info("Bootstrap over %d shared sessions across %d methods", len(ids), len(methods))

    # The real side comes from the same input data for every method; verify and
    # use one canonical copy.
    canonical_method = next(iter(real_by_method))
    canonical_real = real_by_method[canonical_method]
    for method, real in real_by_method.items():
        for sid in ids:
            if real[sid].actions != canonical_real[sid].actions:
                raise ValueError(
                    f"Real session {sid} differs between {canonical_method!r} and "
                    f"{method!r}; methods do not share the same real data"
                )
    real_sessions = [canonical_real[sid] for sid in ids]

    logger.info("Computing intra-session product coherence with SBERT (one-time)")
    semantic = SemanticLevelFidelity()
    semantic._load_sbert()
    real_coherence = semantic._session_coherence_dict(real_sessions)
    real_stats = SessionStats(real_sessions, real_coherence)

    seq = SequenceLevelFidelity()
    synth_stats: dict[str, SessionStats] = {}
    for method in methods:
        synth_sessions = [synth_by_method[method][sid] for sid in ids]
        per_session_scores = raw[method].get("per_session_scores", {})

        pair_lev = np.empty(len(ids))
        for i, sid in enumerate(ids):
            a = real_sessions[i].actions
            b = synth_sessions[i].actions
            denom = max(len(a), len(b))
            pair_lev[i] = seq._levenshtein(a, b) / denom if denom > 0 else 0.0

        pair_prod_sim = np.array(
            [
                per_session_scores.get(sid, {}).get("product_similarity_score", np.nan)
                for sid in ids
            ],
            dtype=float,
        )

        synth_coherence = semantic._session_coherence_dict(synth_sessions)
        synth_stats[method] = SessionStats(
            synth_sessions, synth_coherence, pair_lev=pair_lev, pair_prod_sim=pair_prod_sim
        )

    return ids, real_stats, synth_stats


def sanity_check(
    tables: dict[str, dict],
    real_stats: SessionStats,
    synth_stats: dict[str, SessionStats],
) -> dict[str, dict[str, float]]:
    """Recompute all bootstrap metrics on the identity index and compare with the
    stored simplified values. Returns the recomputed point estimates."""
    n = len(real_stats.lengths)
    identity = np.arange(n)
    point: dict[str, dict[str, float]] = {}
    for method, stats in synth_stats.items():
        point[method] = compute_metrics(real_stats, stats, identity)
        stored = tables.get(method, {})
        for key in BOOTSTRAP_METRICS:
            got, want = point[method][key], stored.get(key)
            if want is None:
                continue
            tol = 5e-3 if key == "product_coherence_gap" else 1e-6
            marker = "OK" if abs(got - want) <= tol else "MISMATCH"
            level = logging.INFO if marker == "OK" else logging.WARNING
            logger.log(
                level,
                "[sanity %s] %s %s: recomputed=%.6f stored=%.6f",
                marker,
                method,
                key,
                got,
                want,
            )
    return point


def run_bootstrap(
    real_stats: SessionStats,
    synth_stats: dict[str, SessionStats],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return delta samples (Ours - baseline) per baseline per metric."""
    if OURS not in synth_stats:
        raise ValueError(f"Method {OURS!r} is required for the bootstrap comparison")
    baselines = [m for m in METHOD_ORDER if m in synth_stats and m != OURS]
    n = len(real_stats.lengths)
    rng = np.random.default_rng(seed)

    deltas = {
        b: {k: np.empty(n_bootstrap) for k in BOOTSTRAP_METRICS} for b in baselines
    }
    for it in range(n_bootstrap):
        idx = rng.integers(0, n, n)  # ONE shared index set per iteration
        ours = compute_metrics(real_stats, synth_stats[OURS], idx)
        for b in baselines:
            other = compute_metrics(real_stats, synth_stats[b], idx)
            for k in BOOTSTRAP_METRICS:
                deltas[b][k][it] = ours[k] - other[k]
        if (it + 1) % 1000 == 0:
            logger.info("Bootstrap %d/%d", it + 1, n_bootstrap)
    return deltas


def run_permutation_test(
    real_stats: SessionStats,
    synth_stats: dict[str, SessionStats],
    point: dict[str, dict[str, float]],
    n_permutations: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Paired permutation test of H0: Ours and baseline are exchangeable.

    Per iteration one shared coin-flip mask swaps the Ours/baseline labels
    within each session pair; metrics are recomputed on both swapped sides.
    Returns ``{baseline: {metric: {"p_two_sided", "p_one_sided"}}}`` with the
    add-one correction, so p is in [1/(P+1), 1].
    """
    baselines = [m for m in METHOD_ORDER if m in synth_stats and m != OURS]
    n = len(real_stats.lengths)
    identity = np.arange(n)
    # Separate stream from the bootstrap so its results stay seed-stable.
    rng = np.random.default_rng(seed + 1)

    vals_a = {b: {k: np.empty(n_permutations) for k in BOOTSTRAP_METRICS} for b in baselines}
    vals_b = {b: {k: np.empty(n_permutations) for k in BOOTSTRAP_METRICS} for b in baselines}
    for it in range(n_permutations):
        mask = rng.random(n) < 0.5  # ONE shared swap mask per iteration
        for b in baselines:
            stats_a, stats_b = _swap_stats(synth_stats[OURS], synth_stats[b], mask)
            m_a = compute_metrics(real_stats, stats_a, identity)
            m_b = compute_metrics(real_stats, stats_b, identity)
            for k in BOOTSTRAP_METRICS:
                vals_a[b][k][it] = m_a[k]
                vals_b[b][k][it] = m_b[k]
        if (it + 1) % 1000 == 0:
            logger.info("Permutation %d/%d", it + 1, n_permutations)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for b in baselines:
        results[b] = {}
        for k in BOOTSTRAP_METRICS:
            delta_obs = point[OURS][k] - point[b][k]
            delta_star = vals_a[b][k] - vals_b[b][k]
            p_two = (1 + np.sum(np.abs(delta_star) >= abs(delta_obs))) / (n_permutations + 1)

            if k in HIGHER_IS_BETTER:
                p_one = (1 + np.sum(delta_star >= delta_obs)) / (n_permutations + 1)
            elif k in ABS_LOWER_IS_BETTER:
                obs_or = abs(point[OURS][k]) - abs(point[b][k])
                star_or = np.abs(vals_a[b][k]) - np.abs(vals_b[b][k])
                p_one = (1 + np.sum(star_or <= obs_or)) / (n_permutations + 1)
            else:  # LOWER_IS_BETTER
                p_one = (1 + np.sum(delta_star <= delta_obs)) / (n_permutations + 1)

            results[b][k] = {
                "p_two_sided": float(p_two),
                "p_one_sided": float(p_one),
            }
    return results


def write_ci_table(
    deltas: dict[str, dict[str, np.ndarray]],
    point: dict[str, dict[str, float]],
    perm: dict[str, dict[str, dict[str, float]]],
    output: Path,
) -> list[dict]:
    rows = []
    for baseline, per_metric in deltas.items():
        for key in BOOTSTRAP_METRICS:
            samples = per_metric[key]
            ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
            p = perm[baseline][key]
            rows.append(
                {
                    "Method": baseline,
                    "Metric": key,
                    "ours_value": point[OURS][key],
                    "method_value": point[baseline][key],
                    "delta": point[OURS][key] - point[baseline][key],
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "significant_boot_ci": bool(ci_low > 0 or ci_high < 0),
                    "p_perm_two_sided": p["p_two_sided"],
                    "p_perm_one_sided": p["p_one_sided"],
                    "significant_perm": bool(p["p_two_sided"] < 0.05),
                }
            )
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Significance table written to %s", output)
    return rows


def _print_significance_summary(rows: list[dict], ours_label: str = OURS) -> None:
    print(f"\n=== Significant comparisons ({ours_label} vs baseline) ===")
    header = f"{'Method':<20} {'Metric':<26} {'delta':>10} {'boot CI':>8} {'p_perm_2s':>10} {'p_perm_1s':>10}"
    print(header)
    for r in rows:
        if not (r["significant_boot_ci"] or r["significant_perm"] or r["p_perm_one_sided"] < 0.05):
            continue
        print(
            f"{r['Method']:<20} {r['Metric']:<26} {r['delta']:>10.4f} "
            f"{'yes' if r['significant_boot_ci'] else 'no':>8} "
            f"{r['p_perm_two_sided']:>10.4f} {r['p_perm_one_sided']:>10.4f}"
        )


def _print_csv(path: Path) -> None:
    print(f"\n=== {path} ===")
    print(path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate eval_results_simplified.json into a final CSV and "
        "compute paired-bootstrap CIs of (Ours - baseline) deltas."
    )
    parser.add_argument("--results-dir", required=True, help="e.g. results_toy_table_test")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write final_table.csv and bootstrap_ci.csv (default: results-dir)",
    )
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations B")
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Paired permutation test iterations P",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Only write the main table (skips bootstrap and permutation test)",
    )
    parser.add_argument(
        "--post-verify-ours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the post verifier over the Ours runs, add an Ours_postverify "
        "row to the main table, and write bootstrap_ci_post_verify.csv comparing "
        "filtered-Ours vs every baseline on the surviving session ids "
        "(--no-post-verify-ours to disable)",
    )
    parser.add_argument(
        "--pv-config",
        default="conf/base.yaml",
        help="YAML config for the post-verifier LLM (default %(default)s)",
    )
    parser.add_argument("--pv-model", default=None, help="Override the post-verifier llm.model")
    parser.add_argument(
        "--pv-threshold",
        type=float,
        default=_DEFAULT_POST_VERIFY_THRESHOLD,
        help="Exclude sessions whose mean post-verify score is below this (default %(default)s)",
    )
    parser.add_argument("--pv-concurrency", type=int, default=8, help="Post-verifier concurrency")
    parser.add_argument(
        "--pv-force",
        action="store_true",
        help="Re-verify runs that already have a post_verify_result.json",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = discover_methods(results_dir)
    if not methods:
        raise SystemExit(f"No experiment folders found under {results_dir}")
    logger.info(
        "Methods: %s", {m: str(p) for m, p in methods.items()}
    )
    missing = [m for m in METHOD_ORDER if m not in methods]
    if missing:
        logger.warning("Methods without a folder: %s", missing)

    tables = load_simplified(methods)

    pv_excluded = None
    if args.post_verify_ours:
        if OURS not in methods:
            raise SystemExit(
                f"Post-verify needs an {OURS!r} folder under {results_dir} "
                "(pass --no-post-verify-ours to tabulate without it)"
            )
        pv_excluded = post_verify_ours(
            methods[OURS],
            args.pv_config,
            args.pv_model,
            args.pv_threshold,
            args.pv_force,
            args.pv_concurrency,
        )
        logger.info("Post-verify: %d pooled session(s) excluded in total", len(pv_excluded))

    # The Ours_postverify row needs the per-session stats even under
    # --skip-bootstrap, so load them whenever either consumer wants them.
    ids = real_stats = synth_stats = None
    if pv_excluded is not None or not args.skip_bootstrap:
        ids, real_stats, synth_stats = load_bootstrap_inputs(methods)

    keep = pv_row = None
    if pv_excluded is not None:
        keep = np.array([i for i, sid in enumerate(ids) if sid not in pv_excluded], dtype=int)
        if keep.size == 0:
            raise SystemExit("Post-verify excluded every shared session; nothing to compare")
        pv_metrics = compute_metrics(real_stats, synth_stats[OURS], keep)
        pv_row = (pv_metrics, int(keep.size), len(ids) - int(keep.size))

    main_csv = output_dir / "final_table.csv"
    write_main_table(
        tables, main_csv, shared_n=len(ids) if ids is not None else None, pv_row=pv_row
    )
    _print_csv(main_csv)

    if args.skip_bootstrap:
        return

    point = sanity_check(tables, real_stats, synth_stats)
    deltas = run_bootstrap(real_stats, synth_stats, args.bootstrap, args.seed)
    perm = run_permutation_test(
        real_stats, synth_stats, point, args.permutations, args.seed
    )
    ci_csv = output_dir / "bootstrap_ci.csv"
    rows = write_ci_table(deltas, point, perm, ci_csv)
    _print_csv(ci_csv)
    _print_significance_summary(rows)

    if keep is not None:
        logger.info(
            "Post-verify comparison: %d surviving of %d shared session(s)",
            keep.size,
            len(ids),
        )
        real_pv = _subset_stats(real_stats, keep)
        synth_pv = {m: _subset_stats(s, keep) for m, s in synth_stats.items()}
        identity = np.arange(keep.size)
        point_pv = {m: compute_metrics(real_pv, s, identity) for m, s in synth_pv.items()}
        deltas_pv = run_bootstrap(real_pv, synth_pv, args.bootstrap, args.seed)
        perm_pv = run_permutation_test(
            real_pv, synth_pv, point_pv, args.permutations, args.seed
        )
        pv_csv = output_dir / "bootstrap_ci_post_verify.csv"
        rows_pv = write_ci_table(deltas_pv, point_pv, perm_pv, pv_csv)
        _print_csv(pv_csv)
        _print_significance_summary(rows_pv, ours_label=OURS_PV)


if __name__ == "__main__":
    main()
