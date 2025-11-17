from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class CrossFeats(FeatureBlock):
    """Small whitelist of hand-crafted cross features."""

    name = "cross"

    def __init__(self, whitelist: List[str] | None = None):
        self.whitelist = whitelist or []
        self.interactions: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None
        self.scope_col: str = "user_id"

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "CrossFeats":
        self.scope_col = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        self.interactions = interactions.copy()
        self.interactions["ts"] = pd.to_datetime(self.interactions["ts"])
        self.items = items.copy() if items is not None else pd.DataFrame()
        return self

    def _brand_match(self, query_id: str, item_id: str, ts_query: pd.Timestamp) -> tuple[float, float]:
        if self.interactions is None or self.items is None or "brand" not in self.items.columns:
            return 0.0, 0.0
        try:
            brand = self.items.set_index("item_id").loc[item_id, "brand"]
        except KeyError:
            return 0.0, 0.0
        hist = self.interactions[(self.interactions[self.scope_col] == query_id) & (self.interactions["ts"] <= ts_query)]
        hist = hist.merge(self.items[["item_id", "brand"]], on="item_id", how="left")
        brand_hist = hist[hist["brand"] == brand]
        if brand_hist.empty:
            return 0.0, 0.0
        recency_h = (ts_query - brand_hist["ts"].max()).total_seconds() / 3600
        return 1.0, float(recency_h)

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            base = {"query_id": row.query_id, "item_id": row.item_id}
            brand_match, brand_recency_h = self._brand_match(row.query_id, row.item_id, ts_query)
            if "brand_match * recency_item_h" in self.whitelist:
                base["x_brand_recency"] = float(brand_match * brand_recency_h)
            if "brand_match" in self.whitelist:
                base["x_brand_match"] = brand_match
            # placeholder for price_rank * novelty type interactions
            if "price_rank_cat * novelty_30d" in self.whitelist:
                base["x_price_novelty"] = 0.0
            rows.append(base)

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("cross", CrossFeats)
