"""Zero-parameter baselines for a TSTR arm, plus test-session grouping.

Why this exists: in the TSTR setup the test label space is built by
intersecting with the synthetic training corpus (``preprocess.py``'s ``P``), so
the two arms face the same targets but can bring very different priors to
them. A model trained
on the narrower corpus concentrates its top-k on a set that is mostly valid
answers, which is worth a large HR@k advantage before it has learned anything
about sequences. Reporting each arm's own popularity prior next to its trained
score is the only way to see whether a TSTR gap reflects learning or bookkeeping:
a gap smaller than the prior gap is not a finding.

Used by ``collect_results.py`` (prior rows in the table) and by
``analysis/tstr_diagnosis/reproduce.py``.
"""
import os
import pickle

import numpy as np


def load_split(data_dir, name):
    """Load ``train.txt``/``test.txt`` as ``(sequences, labels)``."""
    with open(os.path.join(data_dir, name), "rb") as f:
        return pickle.load(f)


def item_support(split):
    """Every item id appearing in a split, as prefix items or as labels."""
    seqs, labels = split
    return {i for s in seqs for i in s} | set(labels)


def popularity_topk(train_split, k=10):
    """Top-k item ids by frequency in the training split (prefixes + labels).

    Counts label occurrences too, so an item that only ever appears as a target
    still registers -- prefix expansion puts the last item of each session in the
    labels of its first example and nowhere else.
    """
    seqs, labels = train_split
    counts = {}
    for s in seqs:
        for i in s:
            counts[i] = counts.get(i, 0) + 1
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    # break frequency ties by item id so the ranking is deterministic
    ranked = sorted(counts, key=lambda i: (-counts[i], i))
    return ranked[:k]


def score_ranking(ranking, test_split):
    """HR@k / MRR@k of a fixed item ranking, as per-example vectors.

    Returns ``(hit, rr)`` float arrays aligned with the test examples, so the
    result can be fed to the same bootstrap as a trained model's dump.
    """
    _, labels = test_split
    pos = {item: r for r, item in enumerate(ranking)}
    hit = np.zeros(len(labels))
    rr = np.zeros(len(labels))
    for j, y in enumerate(labels):
        r = pos.get(y)
        if r is not None:
            hit[j] = 1.0
            rr[j] = 1.0 / (r + 1)
    return hit, rr


def popularity_prior(train_dir, test_dir, k=10):
    """``(hit, rr)`` vectors for the top-k-popularity ranker of ``train_dir``."""
    ranking = popularity_topk(load_split(train_dir, "train.txt"), k)
    return score_ranking(ranking, load_split(test_dir, "test.txt"))


def item_counts(train_split):
    """Item -> frequency over prefixes and labels of a training split."""
    seqs, labels = train_split
    counts = {}
    for s in seqs:
        for i in s:
            counts[i] = counts.get(i, 0) + 1
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return counts


def popularity_prior_range(train_dir, test_dir, k=10, max_combos=2000):
    """HR@k of the top-k-popularity ranker, min/max over cutoff tie-breaks.

    A frequency tie straddling rank k makes the "popularity baseline" ambiguous,
    and on this dataset the ambiguity is not academic: items tied at the rank-k
    cutoff compete for the last slots, and when they are test targets the HR
    moves by around a point depending on an arbitrary ordering. Reporting the
    range keeps the prior from looking more precise than it is. ``n_tied`` and
    ``n_slots`` in the result say how much slack there actually was.

    Returns a dict with ``hr_by_id`` (the deterministic lowest-id ranking used by
    ``popularity_prior``), ``hr_min``/``hr_max``, ``n_tied``, and ``n_slots``.
    """
    import itertools

    train = load_split(train_dir, "train.txt")
    test = load_split(test_dir, "test.txt")
    counts = item_counts(train)
    ranked = sorted(counts, key=lambda i: (-counts[i], i))
    by_id = ranked[:k]
    out = {"hr_by_id": float(score_ranking(by_id, test)[0].mean()),
           "n_tied": 0, "n_slots": 0}
    if len(ranked) <= k:
        out["hr_min"] = out["hr_max"] = out["hr_by_id"]
        return out

    cut = counts[ranked[k - 1]]
    strict = [i for i in ranked if counts[i] > cut]
    tied = [i for i in ranked if counts[i] == cut]
    slots = k - len(strict)
    out["n_tied"], out["n_slots"] = len(tied), slots
    if slots <= 0 or slots >= len(tied):
        out["hr_min"] = out["hr_max"] = out["hr_by_id"]
        return out

    n_combos = 1
    for a in range(slots):
        n_combos = n_combos * (len(tied) - a) // (a + 1)
    if n_combos > max_combos:
        # too many tie-breaks to enumerate; bound by best/worst-case membership
        tgt = set(test[1])
        good = [i for i in tied if i in tgt]
        bad = [i for i in tied if i not in tgt]
        cands = [(good + bad)[:slots], (bad + good)[:slots]]
    else:
        cands = [list(c) for c in itertools.combinations(tied, slots)]
    hrs = [float(score_ranking(strict + c, test)[0].mean()) for c in cands]
    out["hr_min"], out["hr_max"] = min(hrs), max(hrs)
    return out


def random_prior(test_dir, n_items=None, k=10, seed=0, n_draws=200):
    """Expected HR@k / MRR@k of a uniformly random top-k, averaged over draws.

    ``n_items`` defaults to the vocabulary size in ``num_node.txt`` minus the
    padding slot. Averaging many draws rather than using the closed form keeps
    this comparable to the other priors (same per-example vector shape).
    """
    test = load_split(test_dir, "test.txt")
    if n_items is None:
        with open(os.path.join(test_dir, "num_node.txt")) as f:
            n_items = int(f.read().strip()) - 1
    rng = np.random.default_rng(seed)
    hit = np.zeros(len(test[1]))
    rr = np.zeros(len(test[1]))
    for _ in range(n_draws):
        ranking = rng.permutation(np.arange(1, n_items + 1))[:k]
        h, r = score_ranking(list(ranking), test)
        hit += h
        rr += r
    return hit / n_draws, rr / n_draws


# ---------------------------------------------------------------------------
# test-example -> session grouping
# ---------------------------------------------------------------------------


def reconstruct_groups(test_split):
    """Recover each test example's source session index from ``test.txt`` alone.

    ``preprocess.py``'s ``process_seqs`` expands a session ``[a,b,c,d]`` into
    ``([a,b,c],d), ([a,b],c), ([a],b)`` -- consecutive, strictly shrinking, and
    ending at length 1. So a session boundary is exactly after each length-1
    prefix. This is the fallback for datasets emitted before ``test_groups.txt``
    existed; the nesting is verified, so a mismatch raises rather than silently
    producing wrong clusters.
    """
    seqs, _ = test_split
    groups, gid = [], 0
    for j, s in enumerate(seqs):
        groups.append(gid)
        if len(s) == 1:
            gid += 1
        elif j + 1 < len(seqs) and len(seqs[j + 1]) >= len(s):
            # next example is not a shorter prefix, so this session ended early
            gid += 1
    # verify: within a group, each example must be a prefix of the previous one
    for j in range(1, len(seqs)):
        if groups[j] == groups[j - 1]:
            prev, cur = seqs[j - 1], seqs[j]
            if len(cur) >= len(prev) or prev[:len(cur)] != list(cur):
                raise ValueError(
                    "cannot reconstruct test session groups: example %d is not a "
                    "prefix of example %d. Regenerate the dataset so it carries "
                    "test_groups.txt." % (j, j - 1))
    return groups


def load_groups(data_dir):
    """Per-test-example session index: ``test_groups.txt`` if present, else rebuilt."""
    path = os.path.join(data_dir, "test_groups.txt")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return list(pickle.load(f))
    groups = reconstruct_groups(load_split(data_dir, "test.txt"))
    print("note: %s has no test_groups.txt; reconstructed %d sessions from "
          "test.txt prefix structure" % (data_dir, len(set(groups))))
    return groups
