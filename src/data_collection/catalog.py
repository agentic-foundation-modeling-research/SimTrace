"""Required product-catalog lookup: slug -> (category, price_bucket, price)."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"product_handle", "product_title", "product_type", "price"}


class ProductCatalog:
    def __init__(self, path: str, n_buckets: int = 5):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"product catalog not found: {source}")

        cat = pd.read_csv(source)
        missing = sorted(REQUIRED_COLUMNS - set(cat.columns))
        if missing:
            raise ValueError(
                f"product catalog {source} is missing required columns: "
                + ", ".join(missing)
            )
        # Preserve the previous extractor's permissive behavior: a product is
        # retained when it has a title, even if its category or price is absent.
        # Those incomplete products simply receive no catalog enrichment.
        cat = cat.dropna(subset=["product_title"])
        cat["price"] = pd.to_numeric(cat["price"], errors="coerce")
        cat = cat[["product_handle", "product_type", "price"]].drop_duplicates(
            subset=["product_handle"]
        )
        self._slug_to_cat = dict(zip(cat["product_handle"], cat["product_type"]))
        self._slug_to_price = dict(zip(cat["product_handle"], cat["price"]))
        self._bins: dict[str, list[float]] = {}
        complete = cat.dropna(subset=["product_type", "price"])
        complete = complete[
            complete["product_type"].astype(str).str.strip().ne("")
        ]
        for category, grp in complete.groupby("product_type"):
            lo, hi = grp["price"].min(), grp["price"].max()
            self._bins[category] = (
                [lo - 0.5, lo + 0.5]
                if lo == hi
                else [lo + (hi - lo) * i / n_buckets for i in range(n_buckets + 1)]
            )

    def lookup(self, slug: str) -> tuple[str, str, str]:
        if not slug or slug not in self._slug_to_cat:
            return "", "", ""
        category = self._slug_to_cat[slug]
        price = self._slug_to_price[slug]
        if (
            pd.isna(category)
            or not str(category).strip()
            or pd.isna(price)
            or category not in self._bins
        ):
            return "", "", ""
        edges = self._bins[category]
        for i, hi in enumerate(edges[1:], start=1):
            if price <= hi:
                return str(category), str(i), str(price)
        return str(category), str(len(edges) - 1), str(price)


def load_catalog(path: str) -> ProductCatalog:
    """Load and validate the required website product catalog."""
    if not path:
        raise ValueError("a product catalog CSV path is required")
    return ProductCatalog(path)
