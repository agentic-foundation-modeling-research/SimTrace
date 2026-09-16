#!/usr/bin/env python3
"""Multimodal product feature extraction for DIMO / MMSBR (TSTR).

Extracts and caches raw (pre-PCA) product embeddings once per shop catalog:

    data_processed/mm_cache/<store_id>/
        text_bert_<style>_<catalog>.npz   BERT pooler_output (768) per product_id
        img_googlenet.npz                 GoogLeNet avgpool features (1024), shared
        pseudo_imgtext_clip.npz           CLIP image embeddings (512), shared
        pseudo_textimg_clip_<style>_<catalog>.npz  CLIP text embeddings (512)
        manifest.json                     extractor info, missing-image products

``style``   text variant fed to BERT / CLIP-text:
    dimo_paper  -> "title vendor product_type"      (DIMO paper setting)
    mmsbr_paper -> "title"                          (MMSBR paper setting)
``catalog`` is ``real`` or ``synth`` (products.{csv,json} versus
synthetic_products.{csv,json}). Product ids and images are shared; catalog text
is kept condition-specific to reproduce the original paper-setting experiment.

The ``clip`` pseudo-modality replaces the paper pipelines (GoogLeNet ImageNet
class names -> BERT, and DALL-E-mini generated image -> GoogLeNet) with the CLIP
image / text encoders. ``mirror`` reuses GoogLeNet / BERT embeddings as the
pseudo matrices -- a zero-download fallback for CPU smoke tests.

CLI (fills the cache for every shop in the config):
    python3 mm_features.py --config conf/tstr_data.yaml [--device auto]
                           [--pseudo clip] [--force]
"""

import argparse
import csv
import json
import os
import sys
from html.parser import HTMLParser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # session_based_recom/
CACHE_ROOT = os.path.join(ROOT, "data_processed", "mm_cache")

TEXT_STYLES = ("dimo_paper", "mmsbr_paper")
CATALOGS = ("real", "synth")

BERT_MODEL = "bert-base-uncased"
CLIP_MODEL = "openai/clip-vit-base-patch32"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def pick_device(arg="auto"):
    import torch
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _TextExtractor(HTMLParser):
    """Strip tags from product_body_html (stdlib-only; no bs4 dependency)."""

    def __init__(self):
        super().__init__()
        self.chunks = []

    def handle_data(self, data):
        if data.strip():
            self.chunks.append(data.strip())


def strip_html(html):
    if not html:
        return ""
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        return html
    return " ".join(p.chunks)


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_catalog(products_json, products_csv):
    """Merge a shop's products.json (image ids) and products.csv (price) into
    one record per product_id (str):
        {product_id, handle, title, vendor, product_type, price,
         description_html, image_ids}
    Pass the ``synthetic_products.{json,csv}`` pair for the synthetic catalog.
    """
    recs = {}
    for p in json.load(open(products_json)):
        pid = str(p["id"])
        recs[pid] = {
            "product_id": pid,
            "handle": p.get("handle") or "",
            "title": p.get("title") or "",
            "vendor": p.get("vendor") or "",
            "product_type": p.get("product_type") or "",
            "price": None,
            "description_html": p.get("description_html") or "",
            "image_ids": [str(im["id"]) for im in (p.get("images") or [])],
        }
    with open(products_csv, newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("product_id") or "").strip()
            if not pid:
                continue
            rec = recs.setdefault(pid, {
                "product_id": pid, "handle": "", "title": "", "vendor": "",
                "product_type": "", "price": None, "description_html": "",
                "image_ids": [],
            })
            rec["handle"] = rec["handle"] or (row.get("product_handle") or "")
            rec["title"] = rec["title"] or (row.get("product_title") or "")
            rec["vendor"] = rec["vendor"] or (row.get("vendor") or "")
            rec["product_type"] = rec["product_type"] or (row.get("product_type") or "")
            rec["description_html"] = rec["description_html"] \
                or (row.get("product_body_html") or "")
            try:
                rec["price"] = float(row.get("price"))
            except (TypeError, ValueError):
                pass
    return recs


def product_text(rec, style):
    if style == "dimo_paper":
        parts = [rec["title"], rec["vendor"], rec["product_type"]]
    elif style == "mmsbr_paper":
        parts = [rec["title"]]
    else:
        raise ValueError("unknown text style %r" % style)
    return " ".join(x for x in parts if x).strip() or rec["handle"] or "product"


def resolve_image(rec, images_dir):
    """Local image path ``{handle}-{image_id}.png`` (first existing image id),
    or None when no image file is present."""
    for iid in rec["image_ids"]:
        path = os.path.join(images_dir, "%s-%s.png" % (rec["handle"], iid))
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Extractors (all no_grad, batched; return {product_id: 1-D float32 array})
# ---------------------------------------------------------------------------


def extract_bert(texts_by_pid, device, batch_size=16):
    """BERT pooler_output (768-d) -- same recipe as both papers' preprocess."""
    import torch
    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT_MODEL)
    bert = BertModel.from_pretrained(BERT_MODEL).to(device).eval()
    pids = list(texts_by_pid)
    out = {}
    with torch.no_grad():
        for i in range(0, len(pids), batch_size):
            chunk = pids[i:i + batch_size]
            inputs = tok([texts_by_pid[p] for p in chunk], return_tensors="pt",
                         padding=True, truncation=True, max_length=200).to(device)
            cls = bert(**inputs).pooler_output.float().cpu().numpy()
            out.update({p: v.astype(np.float32) for p, v in zip(chunk, cls)})
    return out


def _load_rgb(path):
    from PIL import Image
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img


def extract_googlenet(img_paths_by_pid, device, batch_size=32):
    """GoogLeNet penultimate features (1024-d, avgpool output) -- the
    torchvision equivalent of googlenet_pytorch's .extract_features."""
    import torch
    import torchvision.transforms as T
    from torchvision.models import GoogLeNet_Weights, googlenet
    from torchvision.models.feature_extraction import create_feature_extractor
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT).to(device).eval()
    extractor = create_feature_extractor(model, {"flatten": "feat"})
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pids = list(img_paths_by_pid)
    out = {}
    with torch.no_grad():
        for i in range(0, len(pids), batch_size):
            chunk = pids[i:i + batch_size]
            batch = torch.stack([preprocess(_load_rgb(img_paths_by_pid[p]))
                                 for p in chunk]).to(device)
            feats = extractor(batch)["feat"].float().cpu().numpy()
            out.update({p: v.astype(np.float32) for p, v in zip(chunk, feats)})
    return out


def _clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, proc


def extract_clip_image(img_paths_by_pid, device, batch_size=32):
    import torch
    model, proc = _clip(device)
    pids = list(img_paths_by_pid)
    out = {}
    with torch.no_grad():
        for i in range(0, len(pids), batch_size):
            chunk = pids[i:i + batch_size]
            inputs = proc(images=[_load_rgb(img_paths_by_pid[p]) for p in chunk],
                          return_tensors="pt").to(device)
            emb = model.get_image_features(**inputs).float().cpu().numpy()
            out.update({p: v.astype(np.float32) for p, v in zip(chunk, emb)})
    return out


def extract_clip_text(texts_by_pid, device, batch_size=32):
    import torch
    model, proc = _clip(device)
    pids = list(texts_by_pid)
    out = {}
    with torch.no_grad():
        for i in range(0, len(pids), batch_size):
            chunk = pids[i:i + batch_size]
            inputs = proc(text=[texts_by_pid[p] for p in chunk], return_tensors="pt",
                          padding=True, truncation=True).to(device)
            emb = model.get_text_features(**inputs).float().cpu().numpy()
            out.update({p: v.astype(np.float32) for p, v in zip(chunk, emb)})
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _npz_path(store_id, name):
    return os.path.join(CACHE_ROOT, str(store_id), name + ".npz")


def load_npz(store_id, name):
    path = _npz_path(store_id, name)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_npz(store_id, name, emb_by_pid):
    path = _npz_path(store_id, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **emb_by_pid)
    return path


def _update_manifest(store_id, entry):
    path = os.path.join(CACHE_ROOT, str(store_id), "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = json.load(open(path)) if os.path.exists(path) else {}
    manifest.update(entry)
    json.dump(manifest, open(path, "w"), indent=2, sort_keys=True)


def cached(store_id, name, builder, force=False):
    """Return {pid: vec} for ``name``, building via ``builder()`` on miss."""
    if not force:
        hit = load_npz(store_id, name)
        if hit is not None:
            return hit
    emb = builder()
    save_npz(store_id, name, emb)
    print("  cached %s: %d products" % (name, len(emb)))
    return emb


# ---------------------------------------------------------------------------
# Per-shop feature bundle (what preprocess_mm.py consumes)
# ---------------------------------------------------------------------------


def catalog_paths(shop):
    """Real and synthetic catalog file paths for a resolved shop dict."""
    base = os.path.dirname(shop["products"])
    return {
        "real": (os.path.join(base, "products.json"), shop["products"]),
        "synth": (os.path.join(base, "synthetic_products.json"),
                  os.path.join(base, "synthetic_products.csv")),
    }


def images_dir_of(shop):
    return os.path.join(os.path.dirname(shop["products"]),
                        "images", "copyright-preserve")


def get_text_features(shop, style, catalog, device, force=False):
    """BERT pooler embeddings for one (text style, catalog) pair."""
    sid = str(shop["store_id"])
    pj, pc = catalog_paths(shop)[catalog]
    recs = load_catalog(pj, pc)

    def build():
        texts = {pid: product_text(r, style) for pid, r in recs.items()}
        return extract_bert(texts, device)

    return cached(sid, "text_bert_%s_%s" % (style, catalog), build, force)


def get_image_features(shop, device, force=False):
    """GoogLeNet embeddings (shared across catalogs -- identical images).
    Products with no local image are absent from the dict (mean-fill later)."""
    sid = str(shop["store_id"])
    pj, pc = catalog_paths(shop)["real"]
    recs = load_catalog(pj, pc)
    images_dir = images_dir_of(shop)

    def build():
        paths = {pid: resolve_image(r, images_dir) for pid, r in recs.items()}
        missing = sorted(pid for pid, p in paths.items() if p is None)
        paths = {pid: p for pid, p in paths.items() if p}
        _update_manifest(sid, {"googlenet_missing_image": missing,
                               "n_products": len(recs)})
        if missing:
            print("  %d/%d products have no local image (mean-vector fallback)"
                  % (len(missing), len(recs)))
        return extract_googlenet(paths, device)

    return cached(sid, "img_googlenet", build, force)


def get_pseudo_features(shop, pseudo, style, catalog, device, force=False):
    """(imgText, textImg) pseudo-modality embeddings for MMSBR.

    imgText (pseudo text from image) is image-derived -> shared across styles
    and catalogs; textImg (pseudo image from text) is text-derived -> per
    (style, catalog)."""
    sid = str(shop["store_id"])
    pj, pc = catalog_paths(shop)[catalog]
    recs = load_catalog(pj, pc)
    images_dir = images_dir_of(shop)

    if pseudo == "clip":
        def build_img():
            paths = {pid: resolve_image(r, images_dir) for pid, r in recs.items()}
            paths = {pid: p for pid, p in paths.items() if p}
            return extract_clip_image(paths, device)

        def build_txt():
            texts = {pid: product_text(r, style) for pid, r in recs.items()}
            return extract_clip_text(texts, device)

        img_text = cached(sid, "pseudo_imgtext_clip", build_img, force)
        text_img = cached(sid, "pseudo_textimg_clip_%s_%s" % (style, catalog),
                          build_txt, force)
    elif pseudo == "mirror":
        img_text = get_image_features(shop, device, force)
        text_img = get_text_features(shop, style, catalog, device, force)
    else:
        raise ValueError("unknown pseudo extractor %r (clip|mirror)" % pseudo)
    return img_text, text_img


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


def pca_matrix(emb_by_pid, ordered_pids, dim, missing_fill="mean"):
    """(n_node, dim) float32 matrix, row i <-> item id i+1 (= ordered_pids[i]).

    Products absent from ``emb_by_pid`` (e.g. missing image) get the mean of
    the present embeddings BEFORE PCA. PCA (sklearn) reduces to
    min(dim, n, d) components, then zero-pads columns to exactly ``dim`` so
    tiny --sample vocabs still satisfy the methods' fixed 100/64 widths.
    Returns (matrix, n_missing)."""
    from sklearn.decomposition import PCA
    present = [emb_by_pid[p] for p in ordered_pids if p in emb_by_pid]
    if not present:
        raise SystemExit("No product embeddings available for this vocabulary; "
                         "check catalog/product-id alignment.")
    mean = np.mean(np.stack(present), axis=0)
    raw = np.stack([emb_by_pid.get(p, mean) for p in ordered_pids])
    n_missing = sum(1 for p in ordered_pids if p not in emb_by_pid)

    k = min(dim, raw.shape[0], raw.shape[1])
    # random_state pins sklearn's randomized SVD path so identical inputs give
    # reproducible matrices for identical inputs.
    reduced = PCA(n_components=k, random_state=0).fit_transform(
        raw.astype(np.float64))
    mat = np.zeros((raw.shape[0], dim), dtype=np.float32)
    mat[:, :k] = reduced.astype(np.float32)
    assert np.isfinite(mat).all()
    return mat, n_missing


# ---------------------------------------------------------------------------
# CLI: pre-fill the cache for every shop in a TSTR config
# ---------------------------------------------------------------------------


def main():
    sys.path.insert(0, HERE)
    from preprocess import load_tstr_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pseudo", default="clip", choices=["clip", "mirror"])
    ap.add_argument("--styles", default=",".join(TEXT_STYLES),
                    help="comma list of text styles (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild cache entries")
    args = ap.parse_args()

    cfg = load_tstr_config(argparse.Namespace(
        config=args.config, shop=None, store_id=None, real=None, synth=None,
        products=None))
    device = pick_device(args.device)
    print("device: %s" % device)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    for shop in cfg["shops"]:
        print("shop %r (store %s):" % (shop.get("name"), shop["store_id"]))
        get_image_features(shop, device, args.force)
        for catalog in CATALOGS:
            for style in styles:
                get_text_features(shop, style, catalog, device, args.force)
                get_pseudo_features(shop, args.pseudo, style, catalog, device,
                                    args.force)
    print("cache ready under %s" % CACHE_ROOT)


if __name__ == "__main__":
    main()
