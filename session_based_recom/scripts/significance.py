#!/usr/bin/env python3
"""Uncertainty on a TSTR gap: paired cluster bootstrap over test sessions.

The TSTR table reports mean +/- std over three training seeds, which cannot
answer the question the experiment is for -- "does training on synthetic
sessions do as well as training on real ones?". Three seeds give an interval so
wide it overlaps almost anything (RAIN's real->real HR spans a factor of two), and
a std over seeds says nothing about how much of the gap is test-set sampling
noise. This script produces, for each method and metric:

  * the delta against the real->real baseline,
  * a two-sided 95% CI and a one-sided 95% lower confidence bound (LCB),
  * a bootstrap p-value for H0: delta = 0,
  * a NON-INFERIORITY verdict at a stated margin -- LCB > -margin -- which is
    the actual claim of interest, and
  * an EQUIVALENCE (TOST) verdict, CI entirely inside +/-margin, which
    distinguishes "similar" from merely "not proven worse".

Method
------
Resampling is over the 434 test SESSIONS, not the 996 test examples. Prefix
expansion turns one session into several nested examples that share almost all
their context, so resampling examples would treat correlated observations as
independent and produce intervals that are too narrow.

The bootstrap is PAIRED: each draw of sessions scores both arms on the same
drawn set. This is valid because every condition is evaluated on a byte-identical
test.txt, and it removes the large shared "how hard are these sessions" variance,
which is what makes 996 examples informative at all.

Seeds are handled two ways. The default averages each condition's per-example
outcomes over its seeds and bootstraps sessions only, so the estimand is the
3-seed ensemble and the interval covers test-set sampling. ``--resample-seeds``
additionally resamples seeds with replacement, widening the interval to include
training randomness. With only 3 seeds that component is barely identified, so
both are reported and neither should be read as definitive on its own.

Usage
-----
    python3 scripts/significance.py                       # all methods, defaults
    python3 scripts/significance.py --margin 1.0 --json out.json
    python3 scripts/significance.py --resample-seeds

Requires per-example dumps from ``--preds_out`` (see scripts/preds_io.py);
run scripts/tstr.sh, which passes it automatically.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import preds_io
import priors

SBR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUN_RE = re.compile(r"^(?P<method>[a-z0-9-]+)_(?P<cond>.+?)_seed(?P<seed>\d+)$")
METHOD_ORDER = ["narm", "restc", "rain", "dimo-paper", "mmsbr-paper"]
COND_ORDER = ["real_to_real", "synth_to_real"]
METRICS = [("hit5", "HR@5"), ("rr5", "MRR@5"),
           ("hit10", "HR@10"), ("rr10", "MRR@10")]
BASELINE = "real_to_real"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def discover(preds_dir):
    """``{method: {cond: {seed: {'hit10':arr,'rr10':arr,...}}}}`` from *.npz.

    Invalid prediction dumps are skipped rather than aborting the entire run.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(preds_dir, "*.npz"))):
        base = os.path.splitext(os.path.basename(path))[0]
        m = RUN_RE.match(base)
        if not m:
            print("skipping unrecognised dump name: %s" % base)
            continue
        try:
            d = preds_io.load_preds(path)
        except ValueError as e:
            print("skipping invalid prediction dump: %s (%s)" % (base, e))
            continue
        runs.setdefault(m.group("method"), {}).setdefault(
            m.group("cond"), {})[int(m.group("seed"))] = d
    return runs


def stack_seeds(by_seed, metric):
    """(S, N) matrix of per-example scores, seeds in ascending order."""
    seeds = sorted(by_seed)
    mats = [by_seed[s][metric] for s in seeds]
    n = mats[0].size
    for s, arr in zip(seeds, mats):
        if arr.size != n:
            raise SystemExit(
                "test-set size differs across seeds (%d vs %d): the dumps were not "
                "produced against the same test.txt, so they cannot be paired."
                % (arr.size, n))
    return np.vstack(mats), seeds


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _group_sums(mat, groups, n_groups):
    """Per-seed, per-session sums and per-session example counts.

    Returns ``(sums, counts)`` where ``sums`` is (S, G) and ``counts`` is (G,).
    Working in session sums lets a bootstrap draw be evaluated as
    ``sums[:, draw].sum() / counts[draw].sum()`` -- the example-level mean of the
    resampled sessions -- without materialising the resampled examples.
    """
    S = mat.shape[0]
    sums = np.zeros((S, n_groups))
    counts = np.zeros(n_groups)
    np.add.at(counts, groups, 1.0)
    for s in range(S):
        np.add.at(sums[s], groups, mat[s])
    return sums, counts


def paired_bootstrap(mat_a, mat_b, groups, n_boot=10000, seed=0,
                     resample_seeds=False, block=500):
    """Bootstrap distribution of ``mean(a) - mean(b)``, resampling sessions.

    ``mat_a``/``mat_b`` are (S, N) per-seed per-example scores for the two arms
    (S may differ between them). Both arms are scored on the SAME drawn sessions
    in each replicate, which is the pairing. Returns
    ``(deltas, observed_a, observed_b)``.
    """
    groups = np.asarray(groups)
    G = int(groups.max()) + 1
    sa, counts = _group_sums(mat_a, groups, G)
    sb, _ = _group_sums(mat_b, groups, G)

    total = counts.sum()
    obs_a = sa.mean(axis=0).sum() / total
    obs_b = sb.mean(axis=0).sum() / total

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(block, n_boot - done)
        draw = rng.integers(0, G, size=(b, G))          # (b, G) session indices
        denom = counts[draw].sum(axis=1)                # (b,) examples drawn
        if resample_seeds:
            # independent seed resample per arm: training runs are independent
            ia = rng.integers(0, sa.shape[0], size=(b, sa.shape[0]))
            ib = rng.integers(0, sb.shape[0], size=(b, sb.shape[0]))
            ga = sa[ia].mean(axis=1)                    # (b, G)
            gb = sb[ib].mean(axis=1)
            num_a = np.take_along_axis(ga, draw, axis=1).sum(axis=1)
            num_b = np.take_along_axis(gb, draw, axis=1).sum(axis=1)
        else:
            ga = sa.mean(axis=0)                        # (G,)
            gb = sb.mean(axis=0)
            num_a = ga[draw].sum(axis=1)
            num_b = gb[draw].sum(axis=1)
        deltas[done:done + b] = num_a / denom - num_b / denom
        done += b
    return deltas, obs_a, obs_b


def summarise(deltas, obs_a, obs_b, margin):
    """Percentile CI / LCB / p-value / non-inferiority + equivalence verdicts.

    Everything is in percentage points. The p-value is the percentile-bootstrap
    achieved significance level for H0: delta = 0, ``2*min(P(d<=0), P(d>=0))``.
    """
    d = deltas * 100.0
    obs = (obs_a - obs_b) * 100.0
    lo, hi = np.percentile(d, [2.5, 97.5])
    lcb = float(np.percentile(d, 5.0))
    p_le = float(np.mean(d <= 0.0))
    p_ge = float(np.mean(d >= 0.0))
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "delta": float(obs),
        "delta_rel": float(obs / (obs_b * 100.0)) if obs_b else float("nan"),
        "arm": float(obs_a * 100.0),
        "baseline": float(obs_b * 100.0),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "lcb95_one_sided": lcb,
        "se": float(d.std(ddof=1)),
        "p_value": p,
        "margin": float(margin),
        # the claim of interest: synthetic is not worse than real by > margin
        "non_inferior": bool(lcb > -margin),
        # stronger: the whole CI sits inside +/-margin
        "equivalent": bool(lo > -margin and hi < margin),
    }


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * pvals[i]
        running = max(running, min(1.0, val))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def analyse(runs, groups, n_boot, seed, margin, rel_margin, resample_seeds):
    """All (method, metric, condition) contrasts against the real->real baseline."""
    results = []
    for method in [m for m in METHOD_ORDER if m in runs] + \
                  sorted(set(runs) - set(METHOD_ORDER)):
        conds = runs[method]
        if BASELINE not in conds:
            print("%s: no %s dumps, skipping" % (method, BASELINE))
            continue
        others = [c for c in COND_ORDER if c in conds and c != BASELINE]
        others += sorted(set(conds) - set(COND_ORDER) - {BASELINE})
        for metric, label in METRICS:
            base_mat, base_seeds = stack_seeds(conds[BASELINE], metric)
            for cond in others:
                arm_mat, arm_seeds = stack_seeds(conds[cond], metric)
                if arm_mat.shape[1] != base_mat.shape[1]:
                    raise SystemExit(
                        "%s: %s has %d test examples but %s has %d -- cannot pair"
                        % (method, cond, arm_mat.shape[1], BASELINE, base_mat.shape[1]))
                if len(arm_seeds) < 2 or len(base_seeds) < 2:
                    print("%s %s vs %s: only %d/%d seeds; delta reported but "
                          "seed uncertainty is not estimable"
                          % (method, cond, BASELINE, len(arm_seeds), len(base_seeds)))
                deltas, oa, ob = paired_bootstrap(
                    arm_mat, base_mat, groups, n_boot=n_boot, seed=seed,
                    resample_seeds=resample_seeds)
                m = margin if not rel_margin else rel_margin * ob * 100.0
                row = summarise(deltas, oa, ob, m)
                row.update(method=method, condition=cond, metric=metric,
                           metric_label=label, n_examples=int(arm_mat.shape[1]),
                           n_sessions=int(np.max(groups) + 1),
                           seeds_arm=arm_seeds, seeds_baseline=base_seeds)
                results.append(row)
    for row, p in zip(results, holm([r["p_value"] for r in results])):
        row["p_holm"] = p
    return results


def render(results, resample_seeds):
    if not results:
        print("No contrasts to report.")
        return
    r0 = results[0]
    print("\n== TSTR significance: paired cluster bootstrap over test sessions ==")
    print("%d test examples from %d sessions | %s"
          % (r0["n_examples"], r0["n_sessions"],
             "resampling sessions AND training seeds" if resample_seeds
             else "resampling sessions; seed-averaged per example"))
    print("delta = condition - real_to_real, in percentage points. "
          "LCB = one-sided 95% lower bound.")
    print("NI = non-inferior at the margin (LCB > -margin); "
          "EQ = equivalent (whole CI inside +/-margin).\n")
    hdr = ("%-11s %-22s %-7s %8s %8s %18s %8s %8s %8s  %-3s %-3s"
           % ("Method", "Condition", "Metric", "real", "arm", "delta [95% CI]",
              "LCB", "p", "p_holm", "NI", "EQ"))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print("%-11s %-22s %-7s %8.2f %8.2f  %+6.2f [%+6.2f,%+6.2f] %8.2f %8.3f %8.3f  %-3s %-3s"
              % (r["method"], r["condition"], r["metric_label"],
                 r["baseline"], r["arm"], r["delta"], r["ci_lo"], r["ci_hi"],
                 r["lcb95_one_sided"], r["p_value"], r["p_holm"],
                 "yes" if r["non_inferior"] else "NO",
                 "yes" if r["equivalent"] else "no"))
    margins = {r["margin"] for r in results}
    if len(margins) == 1:
        print("\nmargin used: %.2f pp (absolute) for every contrast" % margins.pop())
    else:
        print("\nmargins used (relative to each baseline):")
        for r in results:
            print("  %-11s %-22s %-7s %.2f pp"
                  % (r["method"], r["condition"], r["metric_label"], r["margin"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=os.path.join(SBR, "results", "preds"),
                    help="dir of per-example .npz dumps from --preds_out")
    ap.add_argument("--data", default=os.path.join(SBR, "data_processed", "real"),
                    help="dataset dir supplying test_groups.txt (the session of each "
                         "test example); reconstructed from test.txt if absent")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="non-inferiority margin in percentage points (default 2.0)")
    ap.add_argument("--rel-margin", type=float, default=None, dest="rel_margin",
                    help="instead express the margin as a fraction of the real->real "
                         "score (e.g. 0.1 = within 10%% of baseline)")
    ap.add_argument("--resample-seeds", action="store_true", dest="resample_seeds",
                    help="also resample training seeds, widening the interval to "
                         "cover training randomness (only 3 seeds: very coarse)")
    ap.add_argument("--json", default=None, help="also write results as JSON")
    args = ap.parse_args()

    if not os.path.isdir(args.preds):
        raise SystemExit(
            "No per-example dumps at %s. Run scripts/tstr.sh (it passes "
            "--preds_out), or re-run each method's --mode infer with "
            "--preds_out results/preds/<method>_<cond>_seed<k>.npz" % args.preds)
    runs = discover(args.preds)
    if not runs:
        raise SystemExit("No *.npz dumps found in %s" % args.preds)
    groups = np.asarray(priors.load_groups(args.data))

    # the grouping must describe exactly the examples the dumps scored
    for method, conds in sorted(runs.items()):
        for cond, by_seed in sorted(conds.items()):
            for s, d in sorted(by_seed.items()):
                if d["n"] != groups.size:
                    raise SystemExit(
                        "%s %s seed%d has %d test examples but %s describes %d. The "
                        "dumps and the grouping come from different datasets."
                        % (method, cond, s, d["n"], args.data, groups.size))

    for method, conds in sorted(runs.items()):
        print("%s: %s" % (method, ", ".join(
            "%s x%d seeds" % (c, len(s)) for c, s in sorted(conds.items()))))

    results = analyse(runs, groups, args.n_boot, args.seed, args.margin,
                      args.rel_margin, args.resample_seeds)
    render(results, args.resample_seeds)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print("\nWrote %s" % args.json)


if __name__ == "__main__":
    main()
