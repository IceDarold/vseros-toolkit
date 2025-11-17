from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class UserAgg(FeatureBlock):
    """User-level aggregates (RFM, diversity, brand/category shares)."""

    name = "user_agg"

    def __init__(self, windows_days: List[int] | None = None, top_cats: int = 20, top_brands: int = 20):
        self.windows_days = windows_days or [30]
        self.top_cats = top_cats
        self.top_brands = top_brands
        self.interactions: pd.DataFrame | None = None
        self.scope_col: str = "user_id"

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "UserAgg":
        self.scope_col = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        df = interactions.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if items is not None:
            meta_cols = [c for c in ["brand", "category", "price"] if c in items.columns]
            if meta_cols:
                df = df.merge(items[["item_id", *meta_cols]], on="item_id", how="left")
        df.sort_values([self.scope_col, "ts"], inplace=True)
        self.interactions = df
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        if self.interactions is None:
            raise ValueError("UserAgg.fit must be called first")

        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            hist = self.interactions[(self.interactions[self.scope_col] == row.query_id) & (self.interactions["ts"] <= ts_query)]
            base = {"query_id": row.query_id, "item_id": row.item_id}
            recency = (ts_query - hist["ts"].max()).total_seconds() / 3600 if not hist.empty else np.nan
            base["user_recency_h"] = recency if np.isfinite(recency) else 0.0
            base["user_cat_entropy"] = 0.0
            base["user_cat_herfindahl"] = 0.0
            base["user_brand_entropy"] = 0.0

            for w in self.windows_days:
                start = ts_query - timedelta(days=w)
                hw = hist[hist["ts"] >= start]
                base[f"user_freq_{w}d"] = float(len(hw))
                base[f"user_unique_items_{w}d"] = float(hw["item_id"].nunique()) if not hw.empty else 0.0
                if "price" in hist.columns:
                    base[f"user_avg_price_{w}d"] = float(hw["price"].mean()) if not hw.empty else 0.0

            # diversity metrics on the widest window available
            if not hist.empty:
                if "category" in hist.columns:
                    cat_counts = hist["category"].value_counts(dropna=True)
                    if len(cat_counts) > 0:
                        probs = (cat_counts / cat_counts.sum()).to_numpy()
                        base["user_cat_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
                        base["user_cat_herfindahl"] = float((probs**2).sum())
                if "brand" in hist.columns:
                    brand_counts = hist["brand"].value_counts(dropna=True)
                    if len(brand_counts) > 0:
                        probs = (brand_counts / brand_counts.sum()).to_numpy()
                        base["user_brand_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())

            rows.append(base)

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("user_agg", UserAgg)
