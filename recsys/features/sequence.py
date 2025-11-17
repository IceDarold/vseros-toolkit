from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class SequenceFeats(FeatureBlock):
    name = "sequence"

    def __init__(self, last_k: List[int] | None = None, max_seq_len: int = 100):
        self.last_k = last_k or [5]
        self.max_seq_len = max_seq_len
        self.interactions: pd.DataFrame | None = None
        self.scope_col: str = "user_id"
        self.item_meta: Dict[str, Dict] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "SequenceFeats":
        self.scope_col = "session_id" if schema.query_scope == "session" and "session_id" in interactions.columns else "user_id"
        df = interactions.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if items is not None:
            meta_cols = [c for c in ["brand", "category"] if c in items.columns]
            if meta_cols:
                df = df.merge(items[["item_id", *meta_cols]], on="item_id", how="left")
            self.item_meta = {str(r.item_id): {k: getattr(r, k, None) for k in meta_cols} for r in items.itertuples(index=False)}
        self.interactions = df
        return self

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        if self.interactions is None:
            raise ValueError("SequenceFeats.fit must be called first")

        rows = []
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            hist = self.interactions[(self.interactions[self.scope_col] == row.query_id) & (self.interactions["ts"] <= ts_query)]
            hist = hist.sort_values("ts").tail(self.max_seq_len)

            base = {"query_id": row.query_id, "item_id": row.item_id}
            base["seq_len"] = float(len(hist))
            if len(hist) > 0:
                base["time_since_last_h"] = float((ts_query - hist["ts"].max()).total_seconds() / 3600)
            else:
                base["time_since_last_h"] = 0.0

            if len(hist) > 1:
                gaps = hist["ts"].diff().dt.total_seconds().dropna()
                base["gap_mean_s"] = float(gaps.mean())
                base["gap_std_s"] = float(gaps.std() if len(gaps) > 1 else 0.0)
                span_minutes = max((hist["ts"].max() - hist["ts"].min()).total_seconds() / 60, 1e-3)
                base["tempo_items_per_min"] = float(len(hist) / span_minutes)
            else:
                base["gap_mean_s"] = 0.0
                base["gap_std_s"] = 0.0
                base["tempo_items_per_min"] = 0.0

            meta = self.item_meta.get(str(row.item_id), {}) if self.item_meta else {}
            target_brand = meta.get("brand")
            target_cat = meta.get("category")

            for k in self.last_k:
                recent = hist.tail(k)
                base[f"last_{k}_unique_items"] = float(recent["item_id"].nunique()) if not recent.empty else 0.0
                base[f"last_{k}_candidate_repeats"] = float((recent["item_id"] == row.item_id).sum()) if not recent.empty else 0.0
                if target_brand is not None and "brand" in recent.columns:
                    mask = recent["brand"] == target_brand
                    if mask.any():
                        positions = np.flatnonzero(mask.to_numpy())
                        base[f"dist_last_brand_{k}"] = float(len(recent) - positions.max())
                    else:
                        base[f"dist_last_brand_{k}"] = 0.0
                if target_cat is not None and "category" in recent.columns:
                    mask = recent["category"] == target_cat
                    if mask.any():
                        positions = np.flatnonzero(mask.to_numpy())
                        base[f"dist_last_cat_{k}"] = float(len(recent) - positions.max())
                    else:
                        base[f"dist_last_cat_{k}"] = 0.0

            rows.append(base)

        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("sequence", SequenceFeats)
