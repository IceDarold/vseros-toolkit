from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class PairwiseCore(FeatureBlock):
    """Pairwise user/session × item signals with strict time-safety.

    The block computes recency/frequency style statistics scoped by the current
    query (session or user) and global item popularity windows. All aggregates
    rely solely on interactions with ``ts <= ts_query`` for each pair.
    """

    name = "pairwise_core"

    def __init__(self, windows_days: List[int] | None = None, decay_alpha: float = 0.05):
        self.windows_days = windows_days or [7, 30]
        self.decay_alpha = decay_alpha
        self.interactions: pd.DataFrame | None = None
        self.schema: Schema | None = None
        self.scope_col: str = "user_id"
        self.item_meta: Dict[str, Dict] = {}
        self.item_daily_counts: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.hist_by_scope: Dict[str, pd.DataFrame] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "PairwiseCore":
        self.schema = schema
        self.scope_col = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        df = interactions.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if items is not None:
            meta_cols = [c for c in ["brand", "category", "price"] if c in items.columns]
            if meta_cols:
                df = df.merge(items[["item_id", *meta_cols]], on="item_id", how="left")
        df.sort_values([self.scope_col, "ts"], inplace=True)
        self.interactions = df

        # cache history per scope for fast lookups
        self.hist_by_scope = {k: g.copy() for k, g in df.groupby(self.scope_col)}

        # prepare global item daily cumulative counts for pop/novelty
        day_counts = df.assign(day=df["ts"].dt.floor("D")).groupby(["item_id", "day"]).size().reset_index(name="cnt")
        for item_id, sub in day_counts.groupby("item_id"):
            days = sub["day"].to_numpy()
            cums = sub["cnt"].cumsum().to_numpy()
            self.item_daily_counts[str(item_id)] = (days.astype("datetime64[ns]"), cums.astype(np.int64))

        if items is not None:
            self.item_meta = {
                str(row.item_id): {"brand": row.brand if "brand" in items.columns else None, "category": row.category if "category" in items.columns else None, "price": row.price if "price" in items.columns else None}
                for row in items.itertuples(index=False)
            }
        return self

    def _history(self, query_id: str, ts_query: pd.Timestamp) -> pd.DataFrame:
        if query_id not in self.hist_by_scope:
            return pd.DataFrame(columns=self.interactions.columns if self.interactions is not None else [])
        hist = self.hist_by_scope[query_id]
        return hist[hist["ts"] <= ts_query]

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
            raise ValueError("PairwiseCore.fit must be called before transform")

        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            hist = self._history(row.query_id, ts_query)

            base = {"query_id": row.query_id, "item_id": row.item_id}
            recency_user = (ts_query - hist["ts"].max()).total_seconds() / 3600 if not hist.empty else np.nan
            base["recency_user_h"] = recency_user if np.isfinite(recency_user) else 0.0

            item_hist = hist[hist["item_id"] == row.item_id]
            recency_item = (ts_query - item_hist["ts"].max()).total_seconds() / 3600 if not item_hist.empty else np.nan
            base["recency_item_h"] = recency_item if np.isfinite(recency_item) else 0.0
            base["seen_before"] = float(len(item_hist) > 0)

            meta = self.item_meta.get(str(row.item_id), {}) if self.item_meta else {}
            brand = meta.get("brand")
            category = meta.get("category")
            brand_hist = hist[hist.get("brand", pd.Series(dtype=object)) == brand] if brand is not None else pd.DataFrame(columns=hist.columns)
            cat_hist = hist[hist.get("category", pd.Series(dtype=object)) == category] if category is not None else pd.DataFrame(columns=hist.columns)

            rec_brand = (ts_query - brand_hist["ts"].max()).total_seconds() / 3600 if not brand_hist.empty else np.nan
            rec_cat = (ts_query - cat_hist["ts"].max()).total_seconds() / 3600 if not cat_hist.empty else np.nan
            base["recency_brand_h"] = rec_brand if np.isfinite(rec_brand) else 0.0
            base["recency_cat_h"] = rec_cat if np.isfinite(rec_cat) else 0.0
            base["brand_match"] = float(brand is not None and not brand_hist.empty)
            base["category_match"] = float(category is not None and not cat_hist.empty)

            for w in self.windows_days:
                start = ts_query - timedelta(days=w)
                hist_window = hist[hist["ts"] >= start]
                base[f"user_count_{w}d"] = float(len(hist_window))
                base[f"seen_times_{w}d"] = float(len(item_hist[item_hist["ts"] >= start]))
                base[f"item_pop_{w}d"] = self._item_pop_window(row.item_id, ts_query, w)
                base[f"novelty_{w}d"] = float(-np.log1p(base[f"item_pop_{w}d"])) if base[f"item_pop_{w}d"] > 0 else 0.0
                if brand is not None:
                    base[f"brand_count_{w}d"] = float(len(brand_hist[brand_hist["ts"] >= start]))
                if category is not None:
                    base[f"cat_count_{w}d"] = float(len(cat_hist[cat_hist["ts"] >= start]))

            rows.append(base)

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("pairwise_core", PairwiseCore)
