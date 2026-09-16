#!/usr/bin/env python3
"""Train-Synthetic-Test-Real preprocessing.

Converts paired real clickstreams and synthetic buyer trajectories into
the standard format consumed by all five recommenders:

    data_processed/<name>/
        train.txt          pickle( (sequences, labels) )  -- prefix expanded
        test.txt           pickle( (sequences, labels) )
        all_train_seq.txt  pickle( [full train sessions] )
        stats.json         {n_items, n_train_sessions, n_test_sessions, ...}
        num_node.txt       n_items + 1  (embedding table size, index 0 = pad)
        item_dict.json     raw item key -> integer index (1..N)

A session like ``[a, b, c, d]`` -> ([a,b,c], d), ([a,b], c), ([a], b).

Usage:
        python preprocess.py tstr --config conf/tstr_data.yaml [--out DIR] [--sample N]
        python preprocess.py tstr --shop example --store-id <id> \
            --real <sessions_raw.csv> --synth <dir_or_json> [--products <products.csv>] [--out DIR]

    Item ids are the real catalog ``product_id`` (keyed ``store_id:product_id``).
    The synthetic corpus exposes only a ``/products/<handle>`` slug, mapped to a
    ``product_id`` via the shop's ``products.csv``. Because the synthetic sessions
    ARE a subset of the real sessions (same session ids), the split is by session
    id, not date. The SAME min_freq filter is applied symmetrically to both corpora:
        train ids   = synthetic session ids whose REAL sequence AND whose SYNTHETIC
                      sequence both survive the min_freq / length filter (intersection)
        real train  = those ids' REAL product sequences   -> real
        synth train = those ids' SYNTHETIC product seqs    -> synth
        P           = products in BOTH training corpora (real-train INTERSECT synth-train)
        test        = every OTHER real session, restricted to P, kept if >=2 items

    CAVEAT on P: because P intersects with the synthetic corpus, the test label
    space is bounded by whatever the generator happened to cover, which hands the
    synth arm a large prior advantage (see analysis/tstr_diagnosis/REPORT.md).
    Any TSTR gap must be read against the popularity priors reported by
    scripts/collect_results.py, not in isolation.

    The command emits three sibling dirs under --out (default data_processed):
        real/           baseline (train = real-train, test = real held-out)
        synth/          full synthetic data (used by DIMO/MMSBR emission)
        synth_matched/  synthetic prefix examples subsampled to the real
                                arm's count (used by NARM/RESTC/RAIN)
    All share ONE integer vocabulary and the same held-out real test examples.
    Each dir also carries test_groups.txt -- the source session index of every test
    example -- which scripts/significance.py needs for its cluster bootstrap.
"""

import argparse
import csv
import glob
import json
import operator
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # session_based_recom/
OUT_ROOT = os.path.join(ROOT, "data_processed")

# csv fields such as dom snapshots can be large; lift the field-size limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ---------------------------------------------------------------------------
# Session builders (real clickstream + synthetic buyer-sim trajectories).
#
# Items are keyed ``"{store_id}:{product_id}"`` so multiple shops combine into one
# corpus without collisions, and so a product viewed in both the real and synthetic
# corpus of the same shop yields an *identical* key. The real ``sessions_raw.csv``
# has ``product_id`` directly; the synthetic ``session_data.json`` has only a
# ``/products/<handle>`` slug, mapped to ``product_id`` via the shop's products.csv.
# ---------------------------------------------------------------------------

PRODUCT_RE = re.compile(r"/products/([^/?#]+)")


def normalize_handle(h):
    """Canonicalize a product handle/slug so real and synthetic keys align.

    Lowercase, strip surrounding whitespace, drop any query/fragment and a
    trailing slash. Returns None for empty input.
    """
    if not h:
        return None
    h = str(h).strip().lower()
    h = h.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    return h or None


def load_product_map(csv_paths):
    """products.csv -> {normalized_handle: product_id}.

    ``csv_paths`` is a shop's ``products.csv`` (str) or a list thereof. The real
    ``sessions_raw.csv`` already carries ``product_id``; this map exists so the
    synthetic corpus -- which only exposes a ``/products/<handle>`` slug -- can be
    keyed by the SAME ``product_id`` (item ids are product ids, not slugs). Rows
    missing either column are skipped.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    handle2id = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                h = normalize_handle(row.get("product_handle"))
                pid = (row.get("product_id") or "").strip()
                if h and pid:
                    handle2id[h] = pid
    return handle2id


def _parse_ts(s):
    """Parse a synthetic ISO timestamp -> epoch seconds (float)."""
    s = str(s)
    try:
        return datetime.fromisoformat(s.replace("Z", "")).timestamp()
    except Exception:
        return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def _sessions_from_events(events, store_id):
    """Collapse per-session (ts, key) event lists into ordered, dedup'd clicks.

    events: dict[session_id] -> list[(ts, item_key)]
    Returns (sess_clicks, sess_date) with consecutive duplicate items collapsed.
    """
    sess_clicks, sess_date = {}, {}
    for sid, evs in events.items():
        evs.sort(key=operator.itemgetter(0))
        seq = []
        for _, key in evs:
            if not seq or seq[-1] != key:  # collapse consecutive repeats of same item
                seq.append(key)
        sess_clicks[sid] = seq
        sess_date[sid] = evs[-1][0]
    return sess_clicks, sess_date


def build_real(csv_paths, store_id, product_map=None, view_actions=("detail",),
                       sample=None):
    """Real clickstream: sessions_raw.csv rows -> per-session item sequences.

    Item view = a row whose ``semantic_action`` is in ``view_actions``. The item
    key is ``store_id:product_id`` -- ``sessions_raw.csv`` carries ``product_id``
    directly, so it is used as-is; if a row lacks it, we fall back to mapping its
    ``product_handle`` through ``product_map`` (from products.csv). Rows with no
    resolvable product id are skipped. Sessions keyed ``store_id:session_id``.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    view_actions = set(view_actions)
    product_map = product_map or {}
    events = defaultdict(list)  # store-scoped session id -> [(ts, key)]
    n = 0
    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if sample and n >= sample:
                    break
                n += 1
                if row.get("semantic_action") not in view_actions:
                    continue
                pid = (row.get("product_id") or "").strip()
                if not pid:  # fall back to handle -> product_id
                    pid = product_map.get(normalize_handle(row.get("product_handle")))
                if not pid:
                    continue
                try:
                    ts = int(row["timestamp"]) / 1000.0  # ms epoch
                except (KeyError, ValueError, TypeError):
                    continue
                sid = "%s:%s" % (store_id, row.get("session_id", ""))
                events[sid].append((ts, "%s:%s" % (store_id, pid)))
        if sample and n >= sample:
            break
    return _sessions_from_events(events, store_id)


def _iter_synth_files(sources):
    """Expand synthetic sources (dirs / globs / files) into a sorted file list.

    A directory is recursively globbed for the per-run ``session_data.json`` (the
    canonical one-session-per-run file under ``runs/<ts>_<hash>/``), NOT the
    timestamped top-level ``session_data_*.json`` aggregate dumps -- matching those
    too would double-count sessions.
    """
    if isinstance(sources, str):
        sources = [sources]
    files = []
    for s in sources:
        if os.path.isdir(s):
            files += sorted(glob.glob(os.path.join(s, "**", "session_data.json"),
                                      recursive=True))
        elif any(ch in s for ch in "*?["):
            files += sorted(glob.glob(s, recursive=True))
        else:
            files.append(s)
    return files


def build_synthetic(sources, store_id, product_map, sample=None):
    """Synthetic buyer-sim trajectories -> per-session item sequences.

    ``sources`` is a dir (globbed for per-run ``session_data.json``), a glob, a
    file, or a list thereof (so several run dirs combine into one training set).
    A product view = an event whose ``clicked_url`` (fallback ``url``) matches
    ``/products/<handle>``; the handle is mapped through ``product_map`` to a
    ``product_id`` so item keys are ``store_id:product_id`` (identical to the real
    corpus). Events whose handle is absent from the catalog are dropped and
    counted. Sessions keyed ``store_id:session_id``.
    """
    files = _iter_synth_files(sources)
    if not files:
        raise SystemExit("No synthetic session_data.json found under: %r" % (sources,))
    events = defaultdict(list)
    n = mapped = unmapped = 0
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        if isinstance(data, dict):  # tolerate {"events": [...]} wrappers
            data = data.get("events") or data.get("sessions") or []
        for ev in data:
            if sample and n >= sample:
                break
            n += 1
            m = PRODUCT_RE.search(ev.get("clicked_url") or "") \
                or PRODUCT_RE.search(ev.get("url") or "")
            if not m:
                continue
            handle = normalize_handle(m.group(1))
            pid = product_map.get(handle) if handle else None
            if not pid:  # handle not in the shop catalog -> cannot resolve product_id
                unmapped += 1
                continue
            mapped += 1
            sid = "%s:%s" % (store_id, ev.get("session_id", ""))
            events[sid].append((_parse_ts(ev.get("timestamp")),
                                "%s:%s" % (store_id, pid)))
        if sample and n >= sample:
            break
    print("  synthetic product views: %d mapped, %d unmapped (handle not in catalog)"
          % (mapped, unmapped))
    return _sessions_from_events(events, store_id)


# ---------------------------------------------------------------------------
# Shared backend
# ---------------------------------------------------------------------------


def process_seqs(seqs, dates):
    """Prefix-expand each session into (prefix, label) pairs (SR-GNN style).

    Also returns ``out_groups``: for each emitted example, the index of the
    session it came from. Examples sharing a group are nested prefixes of one
    session and are therefore strongly correlated -- downstream significance
    testing must resample *sessions*, not examples, or it will understate the
    variance (see scripts/significance.py)."""
    out_seqs, labs, out_dates, out_groups = [], [], [], []
    for gid, (seq, date) in enumerate(zip(seqs, dates)):
        for i in range(1, len(seq)):
            labs.append(seq[-i])
            out_seqs.append(seq[:-i])
            out_dates.append(date)
            out_groups.append(gid)
    return out_seqs, labs, out_dates, out_groups


def filter_sessions(sess_clicks, min_freq):
    """Drop length-1 sessions, then items with freq < min_freq, then sessions
    left with < 2 items. Returns a new dict[str] -> list (originals untouched)."""
    sess_clicks = {s: c for s, c in sess_clicks.items() if len(c) > 1}
    counts = defaultdict(int)
    for c in sess_clicks.values():
        for it in c:
            counts[it] += 1
    kept = {}
    for s, c in sess_clicks.items():
        filt = [it for it in c if counts[it] >= min_freq]
        if len(filt) >= 2:
            kept[s] = filt
    return kept


def build_item_dict(*click_dicts):
    """Assign integer ids (1..N) to product keys in first-seen order.

    ``click_dicts`` are ``{session_id: [item_key, ...]}`` dicts scanned in order
    (e.g. real_train, then synth_train, then test) to build ONE shared vocabulary
    spanning every corpus, so the emitted datasets share item ids and num_node."""
    item_dict = {}
    for clicks in click_dicts:
        for seq in clicks.values():
            for it in seq:
                if it not in item_dict:
                    item_dict[it] = len(item_dict) + 1
    return item_dict


def renumber(tra_raw, tra_dates, tes_raw, tes_dates, item_dict=None):
    """Map raw item-key sequences to 1-based integer-id sequences.

    When ``item_dict`` is None, builds the train vocabulary (first-seen order)
    and renumbers train items from 1. When a shared ``item_dict`` is supplied
    (TSTR: same vocabulary across both output dirs), it is used as-is so
    all dirs share ids / num_node / test.txt. Either way it cold-drops items
    outside the vocabulary, drops sequences left with < 2 items, and asserts
    index consistency. Returns (tra_seqs, kept_tra_dates, tes_seqs, kept_tes_dates,
    item_dict) with sequences still UNexpanded (no prefix expansion)."""
    shared = item_dict is not None
    if item_dict is None:
        item_dict = {}
    tra_seqs, kept_tra_dates = [], []
    for seq, d in zip(tra_raw, tra_dates):
        outseq = []
        for it in seq:
            if it not in item_dict:
                if shared:
                    continue  # never grow a shared vocab; drop stray keys
                item_dict[it] = len(item_dict) + 1
            outseq.append(item_dict[it])
        if len(outseq) >= 2:
            tra_seqs.append(outseq)
            kept_tra_dates.append(d)

    tes_seqs, kept_tes_dates = [], []
    for seq, d in zip(tes_raw, tes_dates):
        outseq = [item_dict[it] for it in seq if it in item_dict]  # cold-drop
        if len(outseq) >= 2:
            tes_seqs.append(outseq)
            kept_tes_dates.append(d)

    n_items = len(item_dict)
    if not n_items or not tra_seqs:
        raise SystemExit("No training sessions survived filtering (try a larger --sample).")
    if not tes_seqs:
        raise SystemExit("Empty test split after cold-drop; check split / vocab overlap.")

    # index-consistency asserts (Step 2b): bijective 1..N vocab, all indices in range
    assert len(set(item_dict.values())) == n_items, "item_dict is not injective"
    assert min(item_dict.values()) == 1 and max(item_dict.values()) == n_items
    for seq in tra_seqs:
        assert seq and all(1 <= x <= n_items for x in seq)
    for seq in tes_seqs:
        assert all(1 <= x <= n_items for x in seq)
    return tra_seqs, kept_tra_dates, tes_seqs, kept_tes_dates, item_dict


def cap_examples(tr_x, tr_y, cap_train_examples, cap_seed):
    """Seeded supervision-volume matching after prefix expansion."""
    if cap_train_examples is None or cap_train_examples >= len(tr_x):
        return tr_x, tr_y, None
    import random
    rng = random.Random(cap_seed)
    keep = sorted(rng.sample(range(len(tr_x)), cap_train_examples))
    info = {
        "cap_train_examples": cap_train_examples,
        "cap_seed": cap_seed,
        "n_train_examples_before_cap": len(tr_x),
        "kept_example_indices": keep,
    }
    print("  volume-matched train: %d -> %d examples (seed %d)"
          % (len(tr_x), len(keep), cap_seed))
    return [tr_x[i] for i in keep], [tr_y[i] for i in keep], info


def _emit(tra_raw, tra_dates, tes_raw, tes_dates, out_dir,
          extra_stats=None, item_dict=None, cap_train_examples=None,
          cap_seed=0):
    """Core emitter shared by every dataset.

    Inputs are lists of *raw item-key* sequences (already length/freq filtered)
    plus parallel date lists. Renumbers via ``renumber`` (shared or train-built
    vocabulary), prefix expands, and dumps the standard files.
    Returns (stats, item_dict).

    """
    tra_seqs, kept_tra_dates, tes_seqs, kept_tes_dates, item_dict = renumber(
        tra_raw, tra_dates, tes_raw, tes_dates, item_dict=item_dict)
    n_items = len(item_dict)

    tr_x, tr_y, _, _ = process_seqs(tra_seqs, kept_tra_dates)
    te_x, te_y, _, te_groups = process_seqs(tes_seqs, kept_tes_dates)
    assert all(1 <= y <= n_items for y in tr_y)
    assert all(1 <= y <= n_items for y in te_y)
    tr_x, tr_y, cap_info = cap_examples(
        tr_x, tr_y, cap_train_examples, cap_seed)

    os.makedirs(out_dir, exist_ok=True)
    pickle.dump((tr_x, tr_y), open(os.path.join(out_dir, "train.txt"), "wb"))
    pickle.dump((te_x, te_y), open(os.path.join(out_dir, "test.txt"), "wb"))
    pickle.dump(tra_seqs, open(os.path.join(out_dir, "all_train_seq.txt"), "wb"))
    # per-test-example session index, for cluster-bootstrap significance testing
    pickle.dump(te_groups, open(os.path.join(out_dir, "test_groups.txt"), "wb"))

    avg_len = sum(len(s) for s in tra_seqs) / max(1, len(tra_seqs))
    stats = {
        "n_items": n_items,
        "n_train_sessions": len(tra_seqs),
        "n_test_sessions": len(tes_seqs),
        "n_train_examples": len(tr_x),
        "n_test_examples": len(te_x),
        "avg_session_len": round(avg_len, 3),
        "num_node": n_items + 1,
    }
    if extra_stats:
        stats.update(extra_stats)
    if cap_info:
        stats.update(cap_info)
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "num_node.txt"), "w") as f:
        f.write(str(n_items + 1))
    json.dump(item_dict, open(os.path.join(out_dir, "item_dict.json"), "w"))
    print("Wrote %s" % out_dir)
    print(json.dumps(stats, indent=2))
    return stats, item_dict


# ---------------------------------------------------------------------------
# TSTR driver
# ---------------------------------------------------------------------------


def _shop_of(key):
    """store_id prefix of a ``store_id:handle`` item key."""
    return key.split(":", 1)[0]


def _items_by_shop(clicks):
    """Distinct item keys grouped by store_id prefix."""
    by_shop = defaultdict(set)
    for c in clicks.values():
        for it in c:
            by_shop[_shop_of(it)].add(it)
    return by_shop


def alignment_report(synth_clicks, real_clicks, real_test_clicks, shops, out_dir,
                     min_overlap=0.05):
    """Verify synthetic and real item keys refer to the same products.

    Two signals per shop:
      * catalog overlap = distinct items shared between the full synthetic and
        full real corpus. This is the mispairing guard: 0 overlap almost always
        means the ``real``/``synth`` entries point at different shops (wrong
        ``store_id`` pairing) or the URL parse is broken -> hard abort.
      * real-test coverage = fraction of the real held-out test vocabulary that
        the synthetic training vocabulary covers. Informational only; it can be
        legitimately low (small synth corpus, tiny held-out day) without a bug,
        but a low value means the TSTR test set is mostly cold-dropped, so it is
        surfaced and warned on below ``min_overlap``.

    Writes ``alignment_report.json`` and returns it."""
    synth_items = _items_by_shop(synth_clicks)
    real_items = _items_by_shop(real_clicks)
    test_items = _items_by_shop(real_test_clicks)

    per_shop = []
    all_shared, all_test, all_cov_shared = set(), set(), set()
    for shop in shops:
        sid = str(shop["store_id"])
        s, r, t = synth_items.get(sid, set()), real_items.get(sid, set()), test_items.get(sid, set())
        catalog_shared = s & r
        test_shared = s & t
        cov = len(test_shared) / len(t) if t else 0.0
        print("  shop %-10s: synth=%d, real=%d, catalog_shared=%d | real-test=%d covered=%d (%.1f%%)"
              % (shop.get("name", sid), len(s), len(r), len(catalog_shared),
                 len(t), len(test_shared), 100 * cov))
        per_shop.append({
            "name": shop.get("name", sid), "store_id": sid,
            "synth_items": len(s), "real_items": len(r),
            "catalog_shared": len(catalog_shared),
            "real_test_items": len(t), "real_test_covered": len(test_shared),
            "real_test_coverage": round(cov, 4),
        })
        if s and r and not catalog_shared:
            raise SystemExit(
                "ALIGNMENT ERROR: shop %r has 0 catalog overlap between its synthetic "
                "and real corpora. The real/synth entries almost certainly point at "
                "different shops (check store_id pairing) or the URL parse is broken."
                % shop.get("name", sid))
        all_shared |= catalog_shared
        all_test |= t
        all_cov_shared |= test_shared

    overall_cov = len(all_cov_shared) / len(all_test) if all_test else 0.0
    print("  OVERALL      : catalog_shared=%d | real-test coverage=%.1f%%"
          % (len(all_shared), 100 * overall_cov))
    if overall_cov < min_overlap:
        print("  WARNING: real-test coverage %.1f%% < min_overlap %.1f%% -- the TSTR test "
              "set will be dominated by cold-dropped items; results may be noisy."
              % (100 * overall_cov, 100 * min_overlap), file=sys.stderr)

    report = {
        "per_shop": per_shop,
        "overall": {
            "catalog_shared": len(all_shared),
            "real_test_items": len(all_test),
            "real_test_covered": len(all_cov_shared),
            "real_test_coverage": round(overall_cov, 4),
        },
        "sample_shared_keys": sorted(all_shared)[:10],
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(report, open(os.path.join(out_dir, "alignment_report.json"), "w"), indent=2)
    return report


def training_fidelity_report(real_train, synth_train, out_dir=None):
    """Measure the corpus shift that most directly drives a TSTR gap.

    The arms intentionally use the same session ids, but their item support,
    next-item marginals, lengths, and transitions can still be very different.
    Values near 1 are good for overlap ratios; values near 0 are good for total
    variation. This is diagnostic only: preprocessing never reshapes synthetic
    data using real statistics, which would leak the baseline into TSTR.
    """
    real_items = Counter(it for seq in real_train.values() for it in seq)
    synth_items = Counter(it for seq in synth_train.values() for it in seq)
    real_targets = Counter(it for seq in real_train.values() for it in seq[1:])
    synth_targets = Counter(it for seq in synth_train.values() for it in seq[1:])
    real_trans = Counter(pair for seq in real_train.values()
                         for pair in zip(seq, seq[1:]))
    synth_trans = Counter(pair for seq in synth_train.values()
                          for pair in zip(seq, seq[1:]))

    def tv(a, b):
        keys = set(a) | set(b)
        za, zb = float(sum(a.values())), float(sum(b.values()))
        if not za or not zb:
            return 1.0
        return 0.5 * sum(abs(a[k] / za - b[k] / zb) for k in keys)

    def weighted_overlap(a, b):
        keys = set(a) | set(b)
        den = sum(max(a[k], b[k]) for k in keys)
        return sum(min(a[k], b[k]) for k in keys) / den if den else 0.0

    shared = set(real_items) & set(synth_items)
    report = {
        "paired_train_sessions": len(set(real_train) & set(synth_train)),
        "real_train_examples": sum(len(s) - 1 for s in real_train.values()),
        "synth_train_examples": sum(len(s) - 1 for s in synth_train.values()),
        "real_mean_session_length": round(
            sum(map(len, real_train.values())) / max(1, len(real_train)), 4),
        "synth_mean_session_length": round(
            sum(map(len, synth_train.values())) / max(1, len(synth_train)), 4),
        "real_item_support": len(real_items),
        "synth_item_support": len(synth_items),
        "shared_item_support": len(shared),
        "synth_support_coverage_of_real": round(
            len(shared) / max(1, len(real_items)), 4),
        "real_support_coverage_of_synth": round(
            len(shared) / max(1, len(synth_items)), 4),
        "next_item_total_variation": round(tv(real_targets, synth_targets), 4),
        "transition_weighted_overlap": round(
            weighted_overlap(real_trans, synth_trans), 4),
    }
    warnings = []
    if report["synth_support_coverage_of_real"] < 0.8:
        warnings.append("synthetic item support covers less than 80% of real training support")
    if report["next_item_total_variation"] > 0.2:
        warnings.append("next-item distributions differ by more than 0.20 total variation")
    if report["transition_weighted_overlap"] < 0.5:
        warnings.append("fewer than 50% of transition counts overlap")
    report["warnings"] = warnings

    print("Training fidelity: support coverage=%.1f%%, next-item TV=%.3f, "
          "transition overlap=%.3f"
          % (100 * report["synth_support_coverage_of_real"],
             report["next_item_total_variation"],
             report["transition_weighted_overlap"]))
    for warning in warnings:
        print("  WARNING: %s" % warning, file=sys.stderr)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(report, open(os.path.join(out_dir, "fidelity_report.json"), "w"),
                  indent=2)
    return report


def load_tstr_config(args):
    """Load the shop-pairing config (YAML or single-shop args) and resolve paths.

    Resolves each shop's ``real``/``synth``/``products`` paths in place (against
    ``data_root``, default CWD) so downstream code never re-resolves. Returns the
    cfg dict with ``view_actions`` / ``min_freq`` / ``min_overlap`` normalized."""
    if args.config:
        try:
            import yaml
        except ImportError:
            raise SystemExit("PyYAML required for --config; pip install pyyaml")
        cfg = yaml.safe_load(open(args.config))
    elif args.shop and args.store_id and args.real and args.synth:
        shop = {"name": args.shop, "store_id": args.store_id,
                "real": args.real, "synth": args.synth}
        if args.products:
            shop["products"] = args.products
        cfg = {"shops": [shop]}
    else:
        raise SystemExit("Provide --config, or all of --shop/--store-id/--real/--synth.")

    cfg["view_actions"] = cfg.get("view_actions", ["detail"])
    cfg["min_freq"] = int(cfg.get("min_freq", 5))
    cfg["min_overlap"] = float(cfg.get("min_overlap", 0.05))

    # Relative data paths resolve against `data_root` (default: the current
    # working directory), so invoke from the repo root where data/ and results_*/
    # live. `data_root` itself may be relative to the CWD.
    data_root = cfg.get("data_root", os.getcwd())
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.getcwd(), data_root)

    def resolve(p):
        if isinstance(p, list):
            return [resolve(x) for x in p]
        return p if os.path.isabs(p) else os.path.join(data_root, p)

    def default_products(real_paths):
        """products.csv sibling of the (first) real sessions_raw.csv path."""
        first = real_paths[0] if isinstance(real_paths, list) else real_paths
        return os.path.join(os.path.dirname(first), "products.csv")

    for shop in cfg["shops"]:
        shop["real"] = resolve(shop["real"])
        shop["synth"] = resolve(shop["synth"])
        shop["products"] = resolve(shop["products"]) if shop.get("products") \
            else default_products(shop["real"])
    return cfg


def seqs_dates(clicks, date):
    """Date-sort a {session: raw_key_seq} dict into parallel (seqs, dates) lists."""
    keys = sorted(clicks, key=lambda s: date.get(s, 0.0))
    return [clicks[s] for s in keys], [date.get(s, 0.0) for s in keys]


def build_corpora(cfg, sample=None, align_out=None):
    """Build the shared TSTR corpora from a resolved config.

    Runs the full corpus build -> session-id split -> symmetric min_freq filter
    -> P intersection -> test build -> alignment check -> shared vocabulary.
    Returns a dict with everything the emitters need:
        real_train / synth_train / test : (raw_key_seqs, dates), date-sorted
        item_dict    : {item_key -> 1..N} shared vocabulary
        common_stats : stats shared by every emitted dir
        shops        : the resolved shop dicts (products.csv paths etc.)"""
    view_actions = cfg["view_actions"]
    min_freq = cfg["min_freq"]
    min_overlap = cfg["min_overlap"]
    shops = cfg["shops"]

    # 1. build ALL real sessions + ALL synthetic sessions per shop, both keyed by
    #    store_id:product_id (synthetic handles mapped through products.csv).
    real_all, real_date = {}, {}
    synth_clicks, synth_date = {}, {}
    for shop in shops:
        sid = str(shop["store_id"])
        print("Building shop %r (store %s) ..." % (shop.get("name", sid), sid))
        product_map = load_product_map(shop["products"])
        if not product_map:
            raise SystemExit("Empty product map for shop %r; check products.csv: %r"
                             % (shop.get("name", sid), shop["products"]))
        rc, rd = build_real(shop["real"], sid, product_map=product_map,
                                    view_actions=view_actions, sample=sample)
        sc, sd = build_synthetic(shop["synth"], sid, product_map,
                                         sample=sample)
        print("  real sessions: %d   synthetic sessions: %d" % (len(rc), len(sc)))
        real_all.update(rc)
        real_date.update(rd)
        synth_clicks.update(sc)
        synth_date.update(sd)

    if not real_all:
        raise SystemExit("No real sessions built; check real paths / view_actions.")
    if not synth_clicks:
        raise SystemExit("No synthetic sessions built; check synth paths / URL parse.")

    # 2. session-id split. Synthetic sessions are a SUBSET of the real sessions
    #    (their ids ARE the conditioning real session ids), so the split is by
    #    session id: the synthetic ids form the training set and every OTHER real
    #    session is held out for test (all conditioning ids excluded -> no leak).
    T = set(synth_clicks)
    real_heldout = {s: c for s, c in real_all.items() if s not in T}

    # 3. The SAME min_freq / length filter is applied INDEPENDENTLY to the real and the
    #    synthetic sequences of each candidate id (the synthetic ids, subset to those
    #    that also exist in the real corpus). An id trains only if it survives in BOTH
    #    corpora, so the two training sets hold the IDENTICAL id list and are treated
    #    symmetrically -- they differ only in sequence content (real vs synthetic), not
    #    in which rare items were trimmed. (The synth filter subsumes the old ">=2
    #    products" guard, since filter_sessions already drops <2-item sessions.)
    real_at_synth = {s: real_all[s] for s in synth_clicks if s in real_all}
    synth_at_synth = {s: synth_clicks[s] for s in synth_clicks if s in real_all}
    real_filt = filter_sessions(real_at_synth, min_freq)
    synth_filt = filter_sessions(synth_at_synth, min_freq)
    train_ids = [s for s in real_filt if s in synth_filt]
    if not train_ids:
        raise SystemExit("No session survived min_freq filtering in BOTH the real and "
                         "synthetic corpora; lower min_freq (or check synth/real pairing).")
    real_train_filt = {s: real_filt[s] for s in train_ids}
    synth_train_filt = {s: synth_filt[s] for s in train_ids}

    # P = products observed in both paired training corpora. Restricting test to
    # P guarantees every target was observed under both conditions. Training
    # sequences remain intact: support and length differences are properties of
    # the synthetic generator and are measured in fidelity_report.json.
    real_items = {it for c in real_train_filt.values() for it in c}
    synth_items = {it for c in synth_train_filt.values() for it in c}
    P = real_items & synth_items
    if not P:
        raise SystemExit("Empty real/synth training product intersection; the two "
                         "corpora share no items (lower min_freq or check pairing).")

    # 4. test = held-out real sessions restricted to P (products trained on by BOTH
    #    corpora), kept if >=2 products remain.
    test_clicks = {}
    for s, c in real_heldout.items():
        filt = [it for it in c if it in P]
        if len(filt) >= 2:
            test_clicks[s] = filt
    if not test_clicks:
        raise SystemExit("Empty test set; no held-out real session has >=2 products "
                         "in the real-train product list (lower min_freq?).")
    print("  synth ids: %d | train ids (survive min_freq in real AND synth): %d "
          "| test sessions: %d | |P|: %d"
          % (len(T), len(train_ids), len(test_clicks), len(P)))

    # 4b. alignment sanity check (aborts on a broken pairing). Catalog overlap uses
    #     the full real corpus; real-test coverage = fraction of the test vocabulary
    #     the synthetic training vocabulary covers (the TSTR coverage signal).
    print("Item-ID alignment (catalog overlap + real-test coverage):")
    if align_out:
        # Use the unfiltered held-out corpus here. Passing ``test_clicks`` would
        # make coverage tautologically 100% because that set was already
        # restricted to P above.
        alignment_report(synth_train_filt, real_all, real_heldout, shops, align_out,
                         min_overlap)
    fidelity = training_fidelity_report(real_train_filt, synth_train_filt,
                                        align_out)

    # 5. ONE shared vocabulary over union(real train, synth train, test), so both
    #    output dirs have identical item ids and model output dimensionality.
    item_dict = build_item_dict(real_train_filt, synth_train_filt, test_clicks)

    common_stats = {"n_shops": len(shops), "min_freq": min_freq,
                    "n_train_ids": len(train_ids), "vocab_size": len(item_dict),
                    "fidelity": fidelity}
    return {
        "real_train": seqs_dates(real_train_filt, real_date),
        "synth_train": seqs_dates(synth_train_filt, synth_date),
        "test": seqs_dates(test_clicks, real_date),
        "item_dict": item_dict,
        "common_stats": common_stats,
        "shops": shops,
    }


def run_tstr(argv):
    ap = argparse.ArgumentParser(prog="preprocess.py tstr")
    ap.add_argument("--config", help="YAML config pairing real/synth per shop")
    ap.add_argument("--out", default=OUT_ROOT, help="output root (default data_processed)")
    ap.add_argument("--sample", type=int, default=None,
                    help="limit raw rows/events read per corpus (fast smoke tests)")
    # single-shop shortcut (no config)
    ap.add_argument("--shop", help="shop name (single-shop mode)")
    ap.add_argument("--store-id", help="store id namespace (single-shop mode)")
    ap.add_argument("--real", help="real sessions_raw.csv (single-shop mode)")
    ap.add_argument("--synth", help="synthetic dir/glob/json (single-shop mode)")
    ap.add_argument("--products", help="products.csv catalog (single-shop mode; "
                    "default: sibling of --real)")
    ap.add_argument("--match-seed", type=int, default=0,
                    help="seed for the ID-method synthetic volume match")
    args = ap.parse_args(argv)

    cfg = load_tstr_config(args)
    c = build_corpora(cfg, sample=args.sample,
                      align_out=os.path.join(args.out, "synth"))
    test_raw, test_dates = c["test"]
    real_train_raw, real_train_dates = c["real_train"]
    synth_train_raw, synth_train_dates = c["synth_train"]
    item_dict, common_stats = c["item_dict"], c["common_stats"]

    # 5a. baseline: real-version of the train ids -> real held-out
    print("\n== real (baseline: real -> real) ==")
    real_stats, _ = _emit(
        real_train_raw, real_train_dates, test_raw, test_dates,
        os.path.join(args.out, "real"), item_dict=item_dict,
        extra_stats=dict(common_stats, mode="baseline_real_to_real"))

    # 5b. TSTR: synthetic-version of the SAME train ids -> SAME real held-out
    print("\n== synth (TSTR: synthetic -> real) ==")
    _emit(synth_train_raw, synth_train_dates, test_raw, test_dates,
          os.path.join(args.out, "synth"), item_dict=item_dict,
          extra_stats=dict(common_stats, mode="tstr_synth_to_real"))

    # 5c. ID-method TSTR: match the number of prefix-expanded synthetic
    # examples to the real baseline. DIMO/MMSBR deliberately keep using the
    # unmodified synthetic multimodal directories emitted by preprocess_mm.py.
    print("\n== synth_matched (ID methods: synthetic -> real, "
          "volume matched to %d examples) ==" % real_stats["n_train_examples"])
    matched_stats, _ = _emit(
        synth_train_raw, synth_train_dates, test_raw, test_dates,
        os.path.join(args.out, "synth_matched"), item_dict=item_dict,
        cap_train_examples=real_stats["n_train_examples"],
        cap_seed=args.match_seed,
        extra_stats=dict(common_stats, mode="tstr_synth_to_real",
                         match_synth_volume="subsample",
                         intended_methods=["narm", "restc", "rain"]))
    assert matched_stats["n_train_examples"] == real_stats["n_train_examples"], \
        "volume-matched synthetic example count differs from real"
    for name in ("test.txt", "test_groups.txt", "item_dict.json", "num_node.txt"):
        real_path = os.path.join(args.out, "real", name)
        matched_path = os.path.join(args.out, "synth_matched", name)
        assert open(real_path, "rb").read() == open(matched_path, "rb").read(), \
            "%s differs between real and matched synthetic directories" % name

def main():
    if len(sys.argv) <= 1 or sys.argv[1] != "tstr":
        raise SystemExit("usage: preprocess.py tstr --config conf/tstr_data.yaml")
    run_tstr(sys.argv[2:])


if __name__ == "__main__":
    main()
