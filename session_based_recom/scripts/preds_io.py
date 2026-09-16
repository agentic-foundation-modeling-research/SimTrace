"""Shared reader/writer for per-example test outcomes (``--preds_out``).

Every method's ``--mode infer`` prints only aggregate Recall / MRR, which
is not enough to put uncertainty on a TSTR gap: with 996 test examples drawn
from 434 sessions and three training seeds, the honest question is whether the
synth-vs-real delta is distinguishable from zero at all. That needs the
*per-example* outcomes so ``scripts/significance.py`` can run a paired cluster
bootstrap over test sessions.

Each infer run writes one ``.npz`` holding

    hit10  float32 (N,)  1.0 if the target is in the top-10, else 0.0
    rr10   float32 (N,)  reciprocal rank of the target, 0.0 if outside the top-10
    hit5   float32 (N,)  1.0 if the target is in the top-5, else 0.0
    rr5    float32 (N,)  reciprocal rank of the target, 0.0 if outside the top-5
    n      int           N
    test_dir str         the test dir the run was evaluated against

in ``test.txt`` order, so runs sharing a test set are aligned elementwise and
can be paired. ``hit10.mean()`` / ``rr10.mean()`` reproduce the run's ``Test:``
line. @10 is the deepest reported cutoff, so it is also the deepest stored: @5
is a strict subset and ``load_preds`` can derive it from ``rr10`` when a dump
predates the @5 keys (rr = 1/rank, so rank <= 5 iff rr10 >= 1/5).

"""
import os

import numpy as np


def dump_preds(path, hit10, rr10, hit5=None, rr5=None, test_dir=None, order=None):
    """Write per-example outcomes to ``path`` (an .npz).

    The positional pair is the *deepest* reported cutoff (@10); @5 is optional
    only so a caller that cannot produce it still writes a loadable dump.

    ``order``, when given, holds the test-example index of each entry in the
    metric arrays; the arrays are permuted into canonical 0..N-1 order. Pass it
    whenever the evaluation loop's iteration order is not simply the test-set
    order -- RAIN's batcher (copied by DIMO and MMSBR), for instance, makes its
    final batch a full-size *overlapping* window when N %% batch_size != 0, so
    some examples are scored twice. Duplicate indices are allowed: the FIRST
    occurrence wins. It is an error for any example to be missing entirely.
    """
    arrays = {"hit10": hit10, "rr10": rr10}
    if (hit5 is None) != (rr5 is None):
        raise ValueError("hit5 and rr5 must be given together")
    if hit5 is not None:
        arrays["hit5"] = hit5
        arrays["rr5"] = rr5
    for k in arrays:
        arrays[k] = np.asarray(arrays[k], dtype=np.float32).ravel()
        if arrays[k].shape != arrays["hit10"].shape:
            raise ValueError("%s length mismatch: %d vs %d"
                             % (k, arrays[k].size, arrays["hit10"].size))

    if order is not None:
        order = np.asarray(order, dtype=np.int64).ravel()
        if order.size != arrays["hit10"].size:
            raise ValueError("order length %d != metric length %d"
                             % (order.size, arrays["hit10"].size))
        n = int(order.max()) + 1 if order.size else 0
        # a valid evaluation covers every example at least once; overlapping
        # final batches score some twice -- keep the first occurrence.
        uniq, first = np.unique(order, return_index=True)
        if uniq.size != n:
            raise ValueError(
                "evaluation order does not cover 0..%d: %d entries, %d unique. "
                "Some test examples were never scored." %
                (n - 1, order.size, uniq.size))
        # uniq is sorted, so first[i] is the first scoring of example i
        for k in arrays:
            arrays[k] = arrays[k][first]

    for prefix in ("10", "5") if "hit5" in arrays else ("10",):
        h, r = arrays["hit" + prefix], arrays["rr" + prefix]
        if not np.all((h == 0.0) | (h == 1.0)):
            raise ValueError("hit%s must be 0/1" % prefix)
        if np.any(r < 0.0) or np.any(r > 1.0):
            raise ValueError("rr%s must lie in [0, 1]" % prefix)
        # a hit implies rr > 0 and vice versa
        if not np.array_equal(h > 0, r > 0):
            raise ValueError("hit%s and rr%s disagree on which examples hit"
                             % (prefix, prefix))
    if "hit5" in arrays:
        # top-5 is a subset of top-10 with identical rr where it hits
        if np.any(arrays["hit5"] > arrays["hit10"]):
            raise ValueError("hit5 must imply hit10")
        both = arrays["hit5"] > 0
        if not np.allclose(arrays["rr5"][both], arrays["rr10"][both]):
            raise ValueError("rr5 must equal rr10 on top-5 hits")

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(path, n=np.int64(arrays["hit10"].size),
             test_dir=str(test_dir or ""), **arrays)
    msg = "Wrote per-example predictions -> %s (n=%d, HR@10=%.4f, MRR@10=%.4f" \
        % (path, arrays["hit10"].size, arrays["hit10"].mean(), arrays["rr10"].mean())
    if "hit5" in arrays:
        msg += ", HR@5=%.4f, MRR@5=%.4f" % (arrays["hit5"].mean(),
                                            arrays["rr5"].mean())
    print(msg + ")")


def load_preds(path):
    """Read a dump back as ``{'hit10','rr10','hit5','rr5','n','test_dir'}``.

    Dumps lacking the @5 arrays get them derived exactly from ``rr10``:
    rr = 1/rank, so the target is in the top-5 iff rr10 >= 1/5 (a small epsilon
    guards float32 reciprocals), and rr5 = rr10 on those examples.

    Dumps must use the current @5/@10 schema.
    """
    with np.load(path, allow_pickle=False) as z:
        if "hit10" not in z:
            raise ValueError(
                "%s does not use the required @5/@10 schema (keys: %s). "
                "Re-run inference with scripts/tstr.sh."
                % (path, ", ".join(sorted(z.files))))
        out = {"hit10": z["hit10"].astype(np.float64),
               "rr10": z["rr10"].astype(np.float64),
               "n": int(z["n"]),
               "test_dir": str(z["test_dir"])}
        if "hit5" in z:
            out["hit5"] = z["hit5"].astype(np.float64)
            out["rr5"] = z["rr5"].astype(np.float64)
        else:
            out["hit5"] = (out["rr10"] >= 1.0 / 5 - 1e-6).astype(np.float64)
            out["rr5"] = out["rr10"] * out["hit5"]
    for k in ("hit10", "rr10", "hit5", "rr5"):
        if out[k].size != out["n"]:
            raise ValueError("%s: %s length disagrees with stored n" % (path, k))
    return out
