from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class PriceNovelties(FeatureBlock):
    """Price and novelty related features (time-safe)."""

    name = "price_nov"

    def __init__(self):
        self.items: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None
        self.scope_col: str = "user_id"
        self.price_rank_cat: Dict[str, float] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "PriceNovelties":
        self.scope_col = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        self.items = items.copy() if items is not None else pd.DataFrame()
        df = interactions.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if not self.items.empty and "price" in self.items.columns:
            df = df.merge(self.items[["item_id", "price", "category"]] if "category" in self.items.columns else self.items[["item_id", "price"]], on="item_id", how="left")
            # price rank per category (static)
            if "category" in self.items.columns:
                cat_groups = self.items.dropna(subset=["price"]).groupby("category")
                for _, sub in cat_groups:
                    prices = sub["price"].rank(method="average") / max(len(sub), 1)
                    for iid, rank in zip(sub["item_id"], prices):
                        self.price_rank_cat[str(iid)] = float(rank)
        self.interactions = df
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        has_price = not self.items.empty and "price" in self.items.columns
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            base = {"query_id": row.query_id, "item_id": row.item_id, "price": 0.0}
            if has_price:
                try:
                    price = float(self.items.set_index("item_id").loc[row.item_id, "price"])
                except KeyError:
                    price = 0.0
                base["price"] = price
                base["price_rank_cat"] = self.price_rank_cat.get(str(row.item_id), 0.0)

                # user price profile strictly before ts_query
                if self.interactions is not None:
                    hist = self.interactions[(self.interactions[self.scope_col] == row.query_id) & (self.interactions["ts"] <= ts_query)]
                    hist = hist.dropna(subset=["price"]) if "price" in hist.columns else pd.DataFrame()
                    if not hist.empty:
                        mean_price = hist["price"].mean()
                        std_price = hist["price"].std() if hist.shape[0] > 1 else 0.0
                        base["price_diff_user_mean"] = float(price - mean_price)
                        base["price_z_user"] = float((price - mean_price) / (std_price + 1e-6))
                    else:
                        base["price_diff_user_mean"] = 0.0
                        base["price_z_user"] = 0.0
            rows.append(base)

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("price_nov", PriceNovelties)
