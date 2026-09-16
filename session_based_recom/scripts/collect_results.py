#!/usr/bin/env python3
"""Parse results/*.log files into MRR/HR @5 and @10 tables.

TSTR logs use ``<method>_<train>_to_<test>.log`` (e.g.
  ``narm_real_to_real.log``, ``narm_synth_to_real.log``)
  -> rendered as a baseline-vs-TSTR table (method x condition), so the
  real->real column and the synth->real columns sit side by side.

Each method prints Recall@5 and Recall@10 (== HR@k for single-target next-item
prediction) and the matching MRRs per epoch (validation) and once at the end on
the held-out test set (the ``Test:`` line from ``--mode infer``). We report the
  **test** result: the last ``Test:`` line.

Multiple seeds per cell (``<method>_<cond>_seed<k>.log`` from ``scripts/tstr.sh``)
are averaged: a cell shows ``mean +/- std`` over its seed logs (sample std, n-1).
A single log with no ``_seed<k>`` suffix renders as a bare value (no ``+/-``).

Beyond the raw table this prints two things the raw table cannot say on its own:

* **deltas vs real->real**, per metric, with a bootstrap 95% lower confidence
  bound and p-value when per-example dumps exist (``results/preds/*.npz``, see
  scripts/significance.py). Three seeds' mean +/- std overlap almost everything,
  so the interval is what decides whether a gap is real.
* **each arm's zero-parameter popularity prior.** The test label space is built
  by intersecting with the synthetic corpus (``preprocess.py``'s ``P``), which
  hands the narrower-support arm a large head start; a TSTR gap smaller than the
  gap between the two priors is bookkeeping, not learning. See
  analysis/tstr_diagnosis/REPORT.md.
"""
import argparse
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# the (?![0-9]) guard stops Recall@5 from matching inside Recall@50 (and @10
# inside @100), which would otherwise silently capture a leading "0"
RECALL_RE = re.compile(r"Recall@5(?![0-9])[:\s]*([0-9]*\.?[0-9]+)")
MRR_RE = re.compile(r"MRR@?5(?![0-9])[:\s]*([0-9]*\.?[0-9]+)")
# @10 twins; logs written before the @10 metrics existed simply lack them
RECALL10_RE = re.compile(r"Recall@10(?![0-9])[:\s]*([0-9]*\.?[0-9]+)")
MRR10_RE = re.compile(r"MRR@?10(?![0-9])[:\s]*([0-9]*\.?[0-9]+)")
SEED_RE = re.compile(r"_seed\d+$")  # strip a trailing _seed<k> so all seeds share one cell

METHOD_ORDER = ["narm", "restc", "rain", "dimo-paper", "mmsbr-paper"]
COND_ORDER = ["real_to_real", "synth_to_real"]
BASELINE_COND = "real_to_real"
PRETTY = {"narm": "NARM", "restc": "RESTC", "rain": "RAIN",
          "dimo-paper": "DIMO (paper)",
          "mmsbr-paper": "MMSBR (paper)",
          "real_to_real": "real->real", "synth_to_real": "synth->real"}

# Each method logs its metrics on its own scale: NARM prints fractions
# (0.1919), RAIN, RESTC, DIMO and MMSBR print percentages (47.7912). Keying the
# conversion on the method is exact; the old heuristic "multiply by 100 if
# <= 1.0" silently inflated any genuine sub-1% percentage by 100x.
LOG_SCALE = {"narm": 100.0, "rain": 1.0, "restc": 1.0,
             "dimo-paper": 1.0, "mmsbr-paper": 1.0}
DEFAULT_SCALE = None   # unknown method -> fall back to the magnitude heuristic

# Training directories behind each condition. ID methods use the matched
# synthetic arm; multimodal methods keep their full paper-setting directories.
COND_TRAIN_DIR = {"real_to_real": "real",
                  "synth_to_real": "synth_matched"}


def condition_train_dir(method, condition):
    if condition == "real_to_real":
        return "real"
    if condition == "synth_to_real" and method in ("dimo-paper", "mmsbr-paper"):
        # Standard two-list split with the same full synthetic sequences; the
        # method-specific train files carry extra parallel arrays.
        return "synth"
    return COND_TRAIN_DIR.get(condition, "")


def pct(v, method=None):
    """Normalise a logged metric to percent using the method's known format."""
    scale = LOG_SCALE.get(method, DEFAULT_SCALE)
    if scale is None:
        # unknown method: fall back to the old heuristic, but say so
        return v * 100.0 if v <= 1.0 else v
    return v * scale


def _pair(line):
    """Extract (hr5, mrr5, hr10, mrr10) from a single line, or None if the
    @5 pair is missing. The @10 slots are None on lines that predate them."""
    r = RECALL_RE.search(line)
    m = MRR_RE.search(line)
    if not (r and m):
        return None
    r10 = RECALL10_RE.search(line)
    m10 = MRR10_RE.search(line)
    return (float(r.group(1)), float(m.group(1)),
            float(r10.group(1)) if r10 else None,
            float(m10.group(1)) if m10 else None)


def _pct4(p, method):
    """Normalise a 4-tuple from _pair to percent, passing None through."""
    return tuple(pct(v, method) if v is not None else None for v in p)


def parse_log(path, method=None):
    """Returns (hr5, mrr5, hr10, mrr10) in percent; @10 entries may be None."""
    text = open(path, errors="ignore").read()
    lines = text.splitlines()

    # Prefer the final held-out test result (the `Test:` line from --mode infer).
    # A decoupled train+infer log contains many per-epoch validation lines plus
    # one Test line; the test number is what we want, not the best validation.
    test = [p for ln in lines if "Test:" in ln for p in (_pair(ln),) if p]
    if test:
        return _pct4(test[-1], method)

    # A partial train-only log can still be inspected: report the best validation
    # Recall@5 epoch and its paired metrics.
    best = None
    for ln in lines:
        p = _pair(ln)
        if p and (best is None or p[0] > best[0]):
            best = p
    if best is None:
        return None
    return _pct4(best, method)


def mean_std(vals):
    """Format a list of numbers as 'mean' (n==1) or 'mean±std' (sample std, n-1)."""
    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return "%.2f" % mean
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return "%.2f±%.2f" % (mean, math.sqrt(var))


def _cell(vals):
    """mean_std over the non-None values, or '--' when a metric predates the
    logs (e.g. @10 columns for logs written before the @10 print existed)."""
    vals = [v for v in vals if v is not None]
    return mean_std(vals) if vals else "--"


def render(results, col_order, title):
    """results: {(method, col): [(hr5, mrr5, hr10, mrr10), ...]} (one entry
    per seed log); render a method x col table with mean±std cells."""
    if not results:
        return
    methods = [m for m in METHOD_ORDER if any(k[0] == m for k in results)]
    methods += sorted({k[0] for k in results} - set(METHOD_ORDER))
    cols = [c for c in col_order if any(k[1] == c for k in results)]
    cols += sorted({k[1] for k in results} - set(col_order))

    print("\n== %s ==" % title)
    hdr = "%-14s" % "Method"
    for c in cols:
        hdr += " | %-55s" % (PRETTY.get(c, c))
    print(hdr)
    sub = "%-14s" % ""
    for _ in cols:
        sub += " | %-13s %-13s %-13s %-13s" % ("MRR@5", "HR@5", "MRR@10", "HR@10")
    print(sub)
    print("-" * len(sub))
    for m in methods:
        row = "%-14s" % PRETTY.get(m, m)
        for c in cols:
            runs = results.get((m, c))
            if runs:
                row += " | %-13s %-13s %-13s %-13s" % (
                    _cell([v[1] for v in runs]), _cell([v[0] for v in runs]),
                    _cell([v[3] for v in runs]), _cell([v[2] for v in runs]))
            else:
                row += " | %-13s %-13s %-13s %-13s" % ("-", "-", "-", "-")
        print(row)


def render_deltas(tstr, sig_rows):
    """Delta of each condition against real->real, per metric.

    ``sig_rows`` (from scripts/significance.py) supplies the bootstrap LCB and
    p-value when per-example dumps exist. Without them only the point delta is
    shown -- honest, but not decidable: a delta whose interval straddles zero is
    not evidence either way.
    """
    conds = [c for c in COND_ORDER if c != BASELINE_COND
             and any(k[1] == c for k in tstr)]
    conds += sorted({k[1] for k in tstr} - set(COND_ORDER) - {BASELINE_COND})
    if not conds:
        return
    methods = [m for m in METHOD_ORDER if (m, BASELINE_COND) in tstr]
    methods += sorted({k[0] for k in tstr if (k[0], BASELINE_COND) in tstr}
                      - set(METHOD_ORDER))
    if not methods:
        print("\n(no real->real logs, so no deltas)")
        return

    lookup = {(r["method"], r["condition"], r["metric_label"]): r for r in sig_rows}
    print("\n== Delta vs real->real (percentage points) ==")
    if sig_rows:
        print("LCB = one-sided 95%% lower bound from a paired cluster bootstrap over "
              "test sessions;\np = bootstrap two-sided p for H0: delta = 0. "
              "NI = non-inferior at the %.1f pp margin." % sig_rows[0]["margin"])
    else:
        print("(point estimates only -- no results/preds/*.npz, so no CI or p-value. "
              "Re-run scripts/tstr.sh, which passes --preds_out.)")
    hdr = "%-14s %-24s %-8s %9s %9s %9s %9s %8s %-4s" % (
        "Method", "Condition", "Metric", "real", "arm", "delta", "LCB", "p", "NI")
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        base = tstr.get((m, BASELINE_COND))
        for c in conds:
            arm = tstr.get((m, c))
            if not arm:
                continue
            for idx, label in ((1, "MRR@5"), (0, "HR@5"),
                               (3, "MRR@10"), (2, "HR@10")):
                bvals = [v[idx] for v in base if v[idx] is not None]
                avals = [v[idx] for v in arm if v[idx] is not None]
                r = lookup.get((m, c, label))
                if r:
                    # report the means the bootstrap actually differenced, so the
                    # row is internally consistent. They can differ slightly from
                    # the logged means when a log predates a metric fix (NARM's
                    # logs average batch means; the dumps average examples).
                    print("%-14s %-24s %-8s %9.2f %9.2f %+9.2f %+9.2f %8.3f %-4s"
                          % (PRETTY.get(m, m), PRETTY.get(c, c), label,
                             r["baseline"], r["arm"],
                             r["delta"], r["lcb95_one_sided"], r["p_value"],
                             "yes" if r["non_inferior"] else "NO"))
                    if bvals and avals:
                        b = sum(bvals) / len(bvals)
                        a = sum(avals) / len(avals)
                        if abs(r["baseline"] - b) > 0.05 or abs(r["arm"] - a) > 0.05:
                            print("%-14s %-24s %-8s   (logs say real=%.2f arm=%.2f; "
                                  "the dumps are authoritative)"
                                  % ("", "", "", b, a))
                elif bvals and avals:
                    b = sum(bvals) / len(bvals)
                    a = sum(avals) / len(avals)
                    print("%-14s %-24s %-8s %9.2f %9.2f %+9.2f %9s %8s %-4s"
                          % (PRETTY.get(m, m), PRETTY.get(c, c), label, b, a,
                             a - b, "-", "-", "-"))
                # neither dumps nor logged @10 values: nothing to report


def render_priors(tstr, data_root):
    """Zero-parameter priors: each arm's top-k training popularity, and random.

    This is the interpretive anchor for everything above. Because the test label
    space is the intersection of the two training corpora's item supports, an arm
    whose corpus covers fewer items concentrates its top-k on a set that is
    mostly valid answers -- worth a large HR@k before any sequence modelling.
    """
    try:
        import priors
    except ImportError:
        return
    arms = []
    if any(c == "real_to_real" for _, c in tstr):
        arms.append(("real_to_real", "real_to_real", "real→real"))
    if any(m in ("narm", "restc", "rain") and c == "synth_to_real"
           for m, c in tstr):
        arms.append(("narm", "synth_to_real", "synth→real (ID, vol-matched)"))
    if any(m in ("dimo-paper", "mmsbr-paper") and c == "synth_to_real"
           for m, c in tstr):
        arms.append(("dimo-paper", "synth_to_real", "synth→real (MM, full)"))
    if not arms:
        return
    test_dir = os.path.join(data_root, COND_TRAIN_DIR[BASELINE_COND])
    if not os.path.isdir(test_dir):
        return

    print("\n== Zero-parameter priors on the SAME test set ==")
    print("Any trained-model delta smaller than the gap between these priors "
          "reflects training-corpus\nitem support, not learned sequence structure. "
          "HR range covers ties at the rank-K cutoff.")
    hdr = "%-30s %9s %9s %9s %9s" % ("Prior", "MRR@K", "HR@K", "HR lo", "HR hi")
    print(hdr)
    print("-" * len(hdr))
    for k in (5, 10):
        for method, c, label in arms:
            d = os.path.join(data_root, condition_train_dir(method, c))
            if not os.path.isdir(d):
                continue
            try:
                hit, rr = priors.popularity_prior(d, test_dir, k=k)
                rng_ = priors.popularity_prior_range(d, test_dir, k=k)
            except (OSError, ValueError) as e:
                print("%-30s  (unavailable: %s)" % (c, e))
                continue
            print("%-30s %9.2f %9.2f %9.2f %9.2f"
                  % ("top-%d popularity: %s" % (k, label),
                     100 * rr.mean(), 100 * hit.mean(),
                     100 * rng_["hr_min"], 100 * rng_["hr_max"]))
        try:
            hit, rr = priors.random_prior(test_dir, k=k)
            print("%-30s %9.2f %9.2f %9s %9s"
                  % ("uniform random top-%d" % k, 100 * rr.mean(),
                     100 * hit.mean(), "", ""))
        except (OSError, ValueError):
            pass
    for method, c, label in arms:
        d = os.path.join(data_root, condition_train_dir(method, c))
        if os.path.isdir(d):
            sup = priors.item_support(priors.load_split(d, "train.txt"))
            tr = priors.load_split(d, "train.txt")
            print("  %-28s %d training items, %d training examples"
                  % (label + ":", len(sup), len(tr[0])))


def load_significance(results_dir, data_root, margin):
    """Run the paired bootstrap if per-example dumps exist; else return []."""
    preds_dir = os.path.join(results_dir, "preds")
    if not os.path.isdir(preds_dir) or not glob.glob(os.path.join(preds_dir, "*.npz")):
        return []
    try:
        import numpy as np
        import priors
        import significance
    except ImportError as e:
        print("\n(significance unavailable: %s)" % e)
        return []
    try:
        runs = significance.discover(preds_dir)
        groups = np.asarray(priors.load_groups(
            os.path.join(data_root, COND_TRAIN_DIR[BASELINE_COND])))
        return significance.analyse(runs, groups, n_boot=10000, seed=0,
                                    margin=margin, rel_margin=None,
                                    resample_seeds=False)
    except (OSError, ValueError, SystemExit) as e:
        print("\n(significance failed: %s)" % e)
        return []


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "results"))
    ap.add_argument("--data_root", default=os.path.join(here, "..", "data_processed"),
                    help="for the popularity priors and test-session grouping")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="non-inferiority margin in percentage points")
    ap.add_argument("--no-significance", action="store_true", dest="no_sig",
                    help="skip the bootstrap even if per-example dumps exist")
    args = ap.parse_args()

    # (method, col) -> {seed or None: (hr, mrr)}. Keyed by seed rather than
    # appended to a list because a stale UNSEEDED log (<method>_<cond>.log from an
    # earlier sweep) collapses to the same cell as the seeded ones, and averaging
    # it in silently mixes runs from different sweeps -- a single stale outlier
    # then moves both the mean and the std of a 3-seed cell.
    raw = {}
    for path in sorted(glob.glob(os.path.join(args.results, "*.log"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        seed_m = re.search(r"_seed(\d+)$", stem)
        seed = int(seed_m.group(1)) if seed_m else None
        base = SEED_RE.sub("", stem)   # collapse <name>_seed<k> -> <name>
        if "_" not in base:
            continue
        method, rest = base.split("_", 1)
        parsed = parse_log(path, method)
        if not parsed:
            continue
        if "_to_" not in rest:
            continue
        raw.setdefault((method, rest), {})[seed] = parsed

    def resolve(d):
        """Drop the unseeded log from any cell that also has seeded logs."""
        out, dropped = {}, []
        for key, by_seed in d.items():
            seeded = {s: v for s, v in by_seed.items() if s is not None}
            if seeded and None in by_seed:
                dropped.append("%s_%s" % key)
            use = seeded if seeded else by_seed
            out[key] = [use[s] for s in sorted(use, key=lambda s: (s is None, s))]
        if dropped:
            print("note: ignoring stale unseeded log(s) for %s -- these cells have "
                  "per-seed logs, and mixing a run from a different sweep into the "
                  "mean/std is not a seed average." % ", ".join(sorted(dropped)))
        return out

    tstr = resolve(raw)
    render(tstr, COND_ORDER, "TSTR (baseline vs synthetic)")
    if tstr:
        sig_rows = [] if args.no_sig else load_significance(
            args.results, args.data_root, args.margin)
        render_deltas(tstr, sig_rows)
        render_priors(tstr, args.data_root)
    if not tstr:
        print("No parseable result logs found in", args.results)


if __name__ == "__main__":
    main()
