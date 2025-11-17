from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class ItemAgg(FeatureBlock):
    """Item-level popularity, age, and trend features."""

    name = "item_agg"

    def __init__(self, windows_days: List[int] | None = None):
        self.windows_days = windows_days or [7, 30]
        self.interactions: pd.DataFrame | None = None
        self.item_first_ts: Dict[str, pd.Timestamp] = {}
        self.item_meta: Dict[str, Dict] = {}
        self.item_daily_counts: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "ItemAgg":
        df = interactions.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if items is not None:
            meta_cols = [c for c in ["brand", "category", "price", "age_added"] if c in items.columns]
            if meta_cols:
                df = df.merge(items[["item_id", *meta_cols]], on="item_id", how="left")
            self.item_meta = {str(r.item_id): {k: getattr(r, k, None) for k in meta_cols} for r in items.itertuples(index=False)}
        self.interactions = df
        self.item_first_ts = df.groupby("item_id")["ts"].min().to_dict()

        # daily counts for time-safe popularity
        day_counts = df.assign(day=df["ts"].dt.floor("D")).groupby(["item_id", "day"]).size().reset_index(name="cnt")
        for item_id, sub in day_counts.groupby("item_id"):
            days = sub["day"].to_numpy()
            cums = sub["cnt"].cumsum().to_numpy()
            self.item_daily_counts[str(item_id)] = (days.astype("datetime64[ns]"), cums.astype(np.int64))
        return self

    def _item_pop_window(self, item_id: str, ts_query: pd.Timestamp, window_days: int) -> float:
        key = str(item_id)
        if key not in self.item_daily_counts:
            return 0.0
        days, cums = self.item_daily_counts[key]
        day = np.datetime64(ts_query.floor("D"))
        idx_cur = np.searchsorted(days, day, side="right") - 1
        cur = cums[idx_cur] if idx_cur >= 0 else 0
        past_day = day - np.timedelta64(window_days, "D")
        idx_past = np.searchsorted(days, past_day, side="right") - 1
        past = cums[idx_past] if idx_past >= 0 else 0
        return float(max(cur - past, 0))

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        if self.interactions is None:
            raise ValueError("ItemAgg.fit must be called before transform")

        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            base = {"query_id": row.query_id, "item_id": row.item_id}
            for w in self.windows_days:
                base[f"item_pop_{w}d"] = self._item_pop_window(row.item_id, ts_query, w)
            base["novelty_item"] = float(-np.log1p(base.get("item_pop_30d", base.get("item_pop_7d", 0.0)))) if base.get("item_pop_30d", 0.0) > 0 else 0.0

            first_ts = self.item_first_ts.get(row.item_id)
            if first_ts is not None:
                base["item_age_days"] = float((ts_query - first_ts).days)
            else:
                base["item_age_days"] = 0.0

            # simple trend: short window vs long window ratio
            short = base.get("item_pop_7d", 0.0)
            long = base.get("item_pop_30d", base.get("item_pop_7d", 0.0))
            base["item_trend_ratio"] = float(short / long) if long > 0 else 0.0

            meta = self.item_meta.get(str(row.item_id), {})
            category = meta.get("category")
            brand = meta.get("brand")
            if category is not None:
                cat_mask = (self.interactions.get("category") == category) & (self.interactions["ts"] <= ts_query)
                for w in self.windows_days:
                    start = ts_query - timedelta(days=w)
                    base[f"cat_pop_{w}d"] = float((cat_mask & (self.interactions["ts"] >= start)).sum())
            if brand is not None:
                brand_mask = (self.interactions.get("brand") == brand) & (self.interactions["ts"] <= ts_query)
                for w in self.windows_days:
                    start = ts_query - timedelta(days=w)
                    base[f"brand_pop_{w}d"] = float((brand_mask & (self.interactions["ts"] >= start)).sum())

            rows.append(base)

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("item_agg", ItemAgg)
