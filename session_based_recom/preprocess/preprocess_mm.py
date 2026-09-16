#!/usr/bin/env python3
"""Multimodal dataset emission for DIMO (SIGIR'24) and MMSBR (TKDE'23).

Builds the SAME TSTR corpora as ``preprocess.py tstr`` (shared item
vocabulary, identical test split) and emits, per condition x method:

    data_processed/{real,synth}_{dimo,mmsbr}_paper/

DIMO dirs (contract reverse-engineered from the released datasets):
    train.txt / test.txt   pickle (prefix_seqs, flags, labels, adj_triple)
                           adj_triple = CSR (data, indices, indptr) of the
                           row-normalized item co-occurrence matrix
                           (n_node x n_node, zero diagonal, built from THIS
                           condition's train sessions, identical in both files)
    frequence.txt          pickle [pos_item, neg_item, pos_weight] each
                           (n_node, 10); pos_item = top-10 co-occurring item
                           ids (1-based, 0-padded), pos_weight = -count,
                           neg_item = sampled zero-co-occurrence items
    textMatrixpca100.npy   (n_node, 100) BERT pooler -> PCA; row i <-> id i+1

MMSBR dirs (contract from the official preprocess3.ipynb + util.py):
    train.txt / test.txt   pickle (seqs, price_seqs, cate_seqs, price_list,
                           cate_list, labels); price/cate ids 1-based,
                           renumbered first-seen in item-id order;
                           price_list/cate_list length n_node, index i <-> id i+1
    imgMatrixpca.npy       (n_node, 64) GoogLeNet -> PCA
    textMatrixpca.npy      (n_node, 64) BERT -> PCA
    imgTextMatrixpca.npy   (n_node, 64) pseudo text-from-image (CLIP image enc)
    textImgMatrixpca.npy   (n_node, 64) pseudo image-from-text (CLIP text enc)
    mm_meta.json           {n_node, n_price, n_category}

Only the published paper text setting is emitted: DIMO uses title + vendor +
product type; MMSBR uses title. The real arm embeds the real catalog, while the
synthetic arm embeds the synthetic catalog that the simulated buyers saw.
Images are shared across catalogs.

CLI (run from the repo root, like preprocess.py):
    python3 preprocess_mm.py --config conf/tstr_data.yaml \
        [--methods dimo,mmsbr] [--out data_processed] \
        [--sample N] [--device auto] [--pseudo clip|mirror] [--force]
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from preprocess import (OUT_ROOT, build_corpora, load_tstr_config,
                        process_seqs, renumber)
import mm_features as mmf

METHODS = ("dimo", "mmsbr")
PAPER_STYLE = {"dimo": "dimo_paper", "mmsbr": "mmsbr_paper"}
TOPK = 10          # frequence.txt column count (upstream fixed)
NEG_SEED = 1234    # seeded neg_item sampling for reproducibility
PRICE_LEVELS = 99  # MMSBR per-category price bins (upstream price_level_num)


def ordered_item_keys(item_dict):
    """Item keys ("store:pid") ordered by id: index i <-> item id i+1."""
    inv = [None] * len(item_dict)
    for key, idx in item_dict.items():
        inv[idx - 1] = key
    assert all(k is not None for k in inv)
    return inv


def merge_by_key(shops, per_shop_getter):
    """{item_key: vec} across shops from a per-shop {pid: vec} getter."""
    out = {}
    for shop in shops:
        sid = str(shop["store_id"])
        for pid, vec in per_shop_getter(shop).items():
            out["%s:%s" % (sid, pid)] = vec
    return out


def write_common(out_dir, tra_seqs, te_groups, item_dict, stats):
    """The standard files every dataset dir carries (mirrors preprocess._emit)."""
    pickle.dump(tra_seqs, open(os.path.join(out_dir, "all_train_seq.txt"), "wb"))
    # per-test-example session index, for cluster-bootstrap significance testing
    pickle.dump(te_groups, open(os.path.join(out_dir, "test_groups.txt"), "wb"))
    with open(os.path.join(out_dir, "num_node.txt"), "w") as f:
        f.write(str(len(item_dict) + 1))
    json.dump(item_dict, open(os.path.join(out_dir, "item_dict.json"), "w"))
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=2)
    print("Wrote %s" % out_dir)
    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# DIMO
# ---------------------------------------------------------------------------


def cooccurrence_counts(train_seqs, n_node):
    """Symmetric item co-occurrence counts over UNexpanded train sessions
    (every position pair within a session counts once per direction),
    zero diagonal."""
    counts = np.zeros((n_node, n_node), dtype=np.float64)
    for seq in train_seqs:
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                i, j = seq[a] - 1, seq[b] - 1
                if i == j:
                    continue
                counts[i, j] += 1
                counts[j, i] += 1
    return counts


def dimo_graph(counts):
    """(adj_triple, pos_item, neg_item, pos_weight) from raw counts.

    adj_triple: CSR (data, indices, indptr) of the row-normalized counts.
    pos_item:   top-10 co-occurring item ids per item (1-based, 0-padded).
    pos_weight: -count aligned with pos_item (0-padded).
    neg_item:   10 zero-co-occurrence item ids (seeded sample)."""
    n = counts.shape[0]
    pos_item = np.zeros((n, TOPK), dtype=np.int64)
    pos_weight = np.zeros((n, TOPK), dtype=np.float64)
    neg_item = np.zeros((n, TOPK), dtype=np.int64)
    rng = np.random.RandomState(NEG_SEED)
    for i in range(n):
        row = counts[i]
        nz = np.flatnonzero(row)
        top = nz[np.argsort(-row[nz], kind="stable")][:TOPK]
        pos_item[i, :len(top)] = top + 1
        pos_weight[i, :len(top)] = -row[top]
        zeros = np.flatnonzero(row == 0)
        zeros = zeros[zeros != i]
        if len(zeros):
            neg = rng.choice(zeros, TOPK, replace=len(zeros) < TOPK)
            neg_item[i] = neg + 1
        # else: leave 0s (pad row) -- only possible in degenerate tiny vocabs

    rowsum = counts.sum(axis=1, keepdims=True)
    adj = np.divide(counts, rowsum, out=np.zeros_like(counts),
                    where=rowsum > 0)
    csr = csr_matrix(adj)
    return (csr.data, csr.indices, csr.indptr), pos_item, neg_item, pos_weight


def dimo_flags(seqs, labels, pos_item, train):
    """Verified upstream rule: flag=+1 iff the label appears in the union of
    the session items' pos_item (top-10 co-occurrence) rows; else -1 in
    train, 0 in test."""
    flags = []
    for seq, lab in zip(seqs, labels):
        union = set()
        for it in set(seq):
            union.update(int(x) for x in pos_item[it - 1])
        hit = int(lab) in union
        flags.append(1 if hit else (-1 if train else 0))
    return flags


def emit_dimo(out_dir, tra_seqs, tes_seqs, text_emb, ordered_keys, item_dict,
              extra_stats):
    n_node = len(ordered_keys)
    # Co-occurrence graph is built from the unexpanded training sessions.
    counts = cooccurrence_counts(tra_seqs, n_node)
    adj_triple, pos_item, neg_item, pos_weight = dimo_graph(counts)

    tr_x, tr_y, _, _ = process_seqs(tra_seqs, [0] * len(tra_seqs))
    te_x, te_y, _, te_groups = process_seqs(tes_seqs, [0] * len(tes_seqs))
    tr_flags = dimo_flags(tr_x, tr_y, pos_item, train=True)
    te_flags = dimo_flags(te_x, te_y, pos_item, train=False)

    text_mat, n_missing_text = mmf.pca_matrix(text_emb, ordered_keys, 100)

    # -- sanity asserts (fail loudly on any contract break) -----------------
    data, indices, indptr = adj_triple
    adj = csr_matrix((data, indices, indptr), shape=(n_node, n_node))
    assert adj.shape == (n_node, n_node)
    assert np.isfinite(adj.data).all()
    assert adj.diagonal().sum() == 0, "adjacency diagonal must be zero"
    rowsums = np.asarray(adj.sum(axis=1)).ravel()
    nz_rows = rowsums > 0
    assert np.allclose(rowsums[nz_rows], 1.0, atol=1e-6), \
        "adjacency rows must sum to 1"
    for arr in (pos_item, neg_item):
        assert arr.shape == (n_node, TOPK)
        assert arr.min() >= 0 and arr.max() <= n_node
    assert pos_weight.shape == (n_node, TOPK) and (pos_weight <= 0).all()
    assert text_mat.shape == (n_node, 100) and np.isfinite(text_mat).all()
    assert set(tr_flags) <= {-1, 1} and set(te_flags) <= {0, 1}
    # independent spot re-derivation of the flag rule on a subset
    for k in range(0, min(200, len(tr_x))):
        want = 1 if any(tr_y[k] in pos_item[it - 1] for it in tr_x[k]) else -1
        assert tr_flags[k] == want, "flag rule mismatch at train sample %d" % k
    # ------------------------------------------------------------------------

    os.makedirs(out_dir, exist_ok=True)
    pickle.dump((tr_x, tr_flags, tr_y, adj_triple),
                open(os.path.join(out_dir, "train.txt"), "wb"))
    pickle.dump((te_x, te_flags, te_y, adj_triple),
                open(os.path.join(out_dir, "test.txt"), "wb"))
    pickle.dump([pos_item, neg_item, pos_weight],
                open(os.path.join(out_dir, "frequence.txt"), "wb"))
    np.save(os.path.join(out_dir, "textMatrixpca100.npy"), text_mat)

    stats = {
        "n_items": n_node,
        "n_train_sessions": len(tra_seqs),
        "n_test_sessions": len(tes_seqs),
        "n_train_examples": len(tr_x),
        "n_test_examples": len(te_x),
        "num_node": n_node + 1,
        "n_missing_product_text": n_missing_text,
        "flag_pos_frac_train": round(
            sum(1 for f in tr_flags if f == 1) / max(1, len(tr_flags)), 4),
    }
    stats.update(extra_stats)
    write_common(out_dir, tra_seqs, te_groups, item_dict, stats)
    return te_x, te_y


# ---------------------------------------------------------------------------
# MMSBR
# ---------------------------------------------------------------------------


def build_price_cate_lists(ordered_keys, shops):
    """(price_list, cate_list, n_price, n_category, n_missing_product).

    Built from the REAL catalogs (ids and prices are identical across the
    real/synthetic catalogs by construction) so every MMSBR dir shares the
    same lists. Category = first-seen renumber of product_type in item-id
    order; price = per-category min-max level int((p-min)/(max-min)*99)+1
    (upstream binning), then first-seen renumber; missing products/prices
    fall back to level 1 / their own "" category."""
    recs = {}
    for shop in shops:
        sid = str(shop["store_id"])
        pj, pc = mmf.catalog_paths(shop)["real"]
        for pid, r in mmf.load_catalog(pj, pc).items():
            recs["%s:%s" % (sid, pid)] = r

    cates, prices, n_missing = [], [], 0
    for key in ordered_keys:
        r = recs.get(key)
        if r is None:
            n_missing += 1
            cates.append("")
            prices.append(None)
        else:
            cates.append(r["product_type"] or "")
            prices.append(r["price"])

    # per-category price range over the vocabulary
    by_cate = {}
    for c, p in zip(cates, prices):
        if p is not None:
            by_cate.setdefault(c, []).append(p)
    ranges = {c: (min(v), max(v)) for c, v in by_cate.items()}

    def raw_level(c, p):
        if p is None or c not in ranges:
            return 1
        lo, hi = ranges[c]
        if hi <= lo:
            return 1
        return int((p - lo) / (hi - lo) * PRICE_LEVELS) + 1

    cate_ids, price_ids = {}, {}
    cate_list, price_list = [], []
    for c, p in zip(cates, prices):
        if c not in cate_ids:
            cate_ids[c] = len(cate_ids) + 1
        lvl = raw_level(c, p)
        if lvl not in price_ids:
            price_ids[lvl] = len(price_ids) + 1
        cate_list.append(cate_ids[c])
        price_list.append(price_ids[lvl])
    return price_list, cate_list, len(price_ids), len(cate_ids), n_missing


def emit_mmsbr(out_dir, tra_seqs, tes_seqs, price_list, cate_list, n_price,
               n_category, matrices, ordered_keys, item_dict, extra_stats):
    n_node = len(ordered_keys)
    tr_x, tr_y, _, _ = process_seqs(tra_seqs, [0] * len(tra_seqs))
    te_x, te_y, _, te_groups = process_seqs(tes_seqs, [0] * len(tes_seqs))
    def parallel(seqs, lst):
        return [[lst[i - 1] for i in seq] for seq in seqs]

    tr_price, tr_cate = parallel(tr_x, price_list), parallel(tr_x, cate_list)
    te_price, te_cate = parallel(te_x, price_list), parallel(te_x, cate_list)

    # -- sanity asserts ------------------------------------------------------
    assert len(price_list) == len(cate_list) == n_node
    assert min(price_list) >= 1 and max(price_list) == n_price
    assert min(cate_list) >= 1 and max(cate_list) == n_category
    for seqs, pseqs, cseqs in ((tr_x, tr_price, tr_cate),
                               (te_x, te_price, te_cate)):
        for s, p, c in zip(seqs, pseqs, cseqs):
            assert len(s) == len(p) == len(c)
            assert all(p[j] == price_list[s[j] - 1] for j in range(len(s)))
    for name, mat in matrices.items():
        assert mat.shape == (n_node, 64), (name, mat.shape)
        assert np.isfinite(mat).all(), name
    # ------------------------------------------------------------------------

    os.makedirs(out_dir, exist_ok=True)
    pickle.dump((tr_x, tr_price, tr_cate, price_list, cate_list, tr_y),
                open(os.path.join(out_dir, "train.txt"), "wb"))
    pickle.dump((te_x, te_price, te_cate, price_list, cate_list, te_y),
                open(os.path.join(out_dir, "test.txt"), "wb"))
    for name, mat in matrices.items():
        np.save(os.path.join(out_dir, name + ".npy"), mat)
    json.dump({"n_node": n_node, "n_price": n_price, "n_category": n_category},
              open(os.path.join(out_dir, "mm_meta.json"), "w"))

    stats = {
        "n_items": n_node,
        "n_train_sessions": len(tra_seqs),
        "n_test_sessions": len(tes_seqs),
        "n_train_examples": len(tr_x),
        "n_test_examples": len(te_x),
        "num_node": n_node + 1,
        "n_price": n_price,
        "n_category": n_category,
    }
    stats.update(extra_stats)
    write_common(out_dir, tra_seqs, te_groups, item_dict, stats)
    return te_x, te_y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Emit DIMO / MMSBR multimodal TSTR datasets")
    ap.add_argument("--config", required=True,
                    help="YAML config pairing real/synth per shop")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma list: dimo,mmsbr (default both)")
    ap.add_argument("--out", default=OUT_ROOT,
                    help="output root (default data_processed)")
    ap.add_argument("--sample", type=int, default=None,
                    help="limit raw rows/events read per corpus (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pseudo", default="clip", choices=["clip", "mirror"],
                    help="MMSBR pseudo-modality extractor (mirror = "
                         "no-download smoke fallback)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild cached feature extractions")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad or not methods:
        raise SystemExit("Invalid --methods %r (methods: %s)" % (bad, METHODS))

    cfg = load_tstr_config(argparse.Namespace(
        config=args.config, shop=None, store_id=None, real=None, synth=None,
        products=None))
    c = build_corpora(cfg, sample=args.sample)
    item_dict, shops = c["item_dict"], c["shops"]
    ordered_keys = ordered_item_keys(item_dict)
    test_raw, test_dates = c["test"]

    # shared vocab => the SAME item_dict/test split as {real,synth}
    ref = os.path.join(args.out, "real", "item_dict.json")
    if args.sample is None and os.path.exists(ref):
        if json.load(open(ref)) != item_dict:
            raise SystemExit(
                "item_dict differs from %s -- the id-only dirs are stale; "
                "re-run scripts/prepare_data.sh first (or remove them)."
                % ref)

    conditions = {}
    for cond, (raw, dates) in (("real", c["real_train"]),
                               ("synth", c["synth_train"])):
        tra, _, tes, _, _ = renumber(raw, dates, test_raw, test_dates,
                                     item_dict=item_dict)
        conditions[cond] = (tra, tes)
    assert conditions["real"][1] == conditions["synth"][1], \
        "test sequences must be identical across conditions"

    conds = ["real", "synth"]

    device = mmf.pick_device(args.device)
    print("device: %s | methods: %s | variant: paper | pseudo: %s"
          % (device, methods, args.pseudo))

    if "mmsbr" in methods:
        price_list, cate_list, n_price, n_category, n_missing_prod = \
            build_price_cate_lists(ordered_keys, shops)
        img_emb = merge_by_key(
            shops, lambda s: mmf.get_image_features(s, device, args.force))
        img_mat, n_missing_img = mmf.pca_matrix(img_emb, ordered_keys, 64)

    # Reproduce the original paper-setting experiment: each arm uses the catalog
    # content associated with the sessions on which it trains.
    CATALOG_OF = {"real": "real", "synth": "synth"}
    MODE_OF = {"real": "baseline_real_to_real",
               "synth": "tstr_synth_to_real"}
    test_ref = {}   # method -> (te_x, te_y) for the cross-dir identity check
    text_mats = {}  # (method, variant) -> {cond: matrix} difference check
    for cond in conds:
        catalog = CATALOG_OF[cond]
        tra_seqs, tes_seqs = conditions[cond]
        for method in methods:
            for variant in ("paper",):
                style = PAPER_STYLE[method]
                out_dir = os.path.join(
                    args.out, "%s_%s_%s" % (cond, method, variant))
                print("\n== %s ==" % os.path.basename(out_dir))
                text_emb = merge_by_key(
                    shops, lambda s: mmf.get_text_features(
                        s, style, catalog, device, args.force))
                extra = {
                    "mode": MODE_OF[cond],
                    "method": method,
                    "text_variant": variant,
                    "text_style": style,
                    "catalog": catalog,
                }
                extra.update(c["common_stats"])

                if method == "dimo":
                    te = emit_dimo(out_dir, tra_seqs, tes_seqs, text_emb,
                                   ordered_keys, item_dict, extra)
                    tmat = np.load(
                        os.path.join(out_dir, "textMatrixpca100.npy"))
                else:
                    text_mat, n_missing_text = mmf.pca_matrix(
                        text_emb, ordered_keys, 64)
                    pseudo_img, pseudo_txt = {}, {}
                    for shop in shops:
                        pi, pt = mmf.get_pseudo_features(
                            shop, args.pseudo, style, catalog, device,
                            args.force)
                        sid = str(shop["store_id"])
                        pseudo_img.update(
                            {"%s:%s" % (sid, k): v for k, v in pi.items()})
                        pseudo_txt.update(
                            {"%s:%s" % (sid, k): v for k, v in pt.items()})
                    imgtext_mat, _ = mmf.pca_matrix(pseudo_img, ordered_keys, 64)
                    textimg_mat, _ = mmf.pca_matrix(pseudo_txt, ordered_keys, 64)
                    matrices = {
                        "imgMatrixpca": img_mat,
                        "textMatrixpca": text_mat,
                        "imgTextMatrixpca": imgtext_mat,
                        "textImgMatrixpca": textimg_mat,
                    }
                    extra.update({
                        "n_missing_image": n_missing_img,
                        "n_missing_product": n_missing_prod,
                        "n_missing_product_text": n_missing_text,
                        "pseudo_extractor": args.pseudo,
                    })
                    te = emit_mmsbr(out_dir, tra_seqs, tes_seqs, price_list,
                                    cate_list, n_price, n_category, matrices,
                                    ordered_keys, item_dict, extra)
                    tmat = text_mat

                # cross-dir invariants
                if method in test_ref:
                    assert te == test_ref[method], \
                        "test (seqs, labels) differ across %s dirs" % method
                else:
                    test_ref[method] = te
                text_mats.setdefault((method, variant), {})[cond] = tmat

    # The two catalogs intentionally have different text. Identical projected
    # matrices usually means the wrong catalog was selected or copied.
    for (method, variant), mats in text_mats.items():
        if np.array_equal(mats["real"], mats["synth"]):
            raise AssertionError(
                "%s/%s text matrix is identical across real/synth catalogs"
                % (method, variant))
    print("\nDone: %d dataset dirs."
          % (len(conds) * len(methods)))


if __name__ == "__main__":
    main()
