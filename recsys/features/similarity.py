from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.dataio.schema import Schema
from recsys.features.base import FeatureBlock, register

logger = logging.getLogger(__name__)


class SimilarityFeats(FeatureBlock):
    name = "similarity"

    def __init__(self, use_item2vec: bool = False, use_tfidf: bool = True, profile_last_k: int = 20):
        self.use_item2vec = use_item2vec
        self.use_tfidf = use_tfidf
        self.profile_last_k = profile_last_k
        self.items: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.item_index: Dict[str, int] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None,
        *,
        schema: Schema,
        profile: Dict,
        rng: np.random.RandomState,
    ) -> "SimilarityFeats":
        self.items = items.copy() if items is not None else pd.DataFrame()
        self.interactions = interactions.copy()
        self.interactions["ts"] = pd.to_datetime(self.interactions["ts"])

        if self.use_tfidf and items is not None:
            texts = (items.get("title", "").fillna("") + " " + items.get("text", "").fillna("") if "text" in items.columns else items.get("title", "").fillna(""))
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.tfidf_matrix = normalize(self.tfidf_matrix)
            self.item_index = {str(item): idx for idx, item in enumerate(items["item_id"]) }
        return self

    def _tfidf_vector(self, item_id: str):
        if self.tfidf_matrix is None:
            return None
        idx = self.item_index.get(str(item_id))
        if idx is None:
            return None
        return self.tfidf_matrix[idx]

    def transform(self, pairs: pd.DataFrame, *, schema: Schema, profile: Dict) -> pd.DataFrame:
        if self.interactions is None:
            raise ValueError("SimilarityFeats.fit must be called before transform")

        rows = []
        scope_col = "session_id" if schema.query_scope == "session" else "user_id"
        if scope_col not in self.interactions.columns:
            scope_col = "user_id"
        for row in pairs.itertuples(index=False):
            ts_query = pd.to_datetime(getattr(row, "ts_query", None) or pairs["ts_query"].max())
            hist = self.interactions[(self.interactions[scope_col] == row.query_id) & (self.interactions["ts"] <= ts_query)]
            hist = hist.sort_values("ts").tail(self.profile_last_k)
            base = {"query_id": row.query_id, "item_id": row.item_id}
            base["cat_match_share"] = 0.0
            if not self.items.empty and "category" in self.items.columns:
                try:
                    target_cat = self.items.set_index("item_id").loc[row.item_id, "category"]
                except KeyError:
                    target_cat = None
                if target_cat is not None:
                    hist_items = hist.merge(self.items[["item_id", "category"]], on="item_id", how="left")
                    denom = max(len(hist_items), 1)
                    base["cat_match_share"] = float((hist_items["category"] == target_cat).sum()) / denom

            if self.use_tfidf and self.tfidf_matrix is not None:
                profile_vecs = []
                for iid in hist["item_id"].tolist():
                    vec = self._tfidf_vector(iid)
                    if vec is not None:
                        profile_vecs.append(vec)
                cand_vec = self._tfidf_vector(row.item_id)
                if profile_vecs and cand_vec is not None:
                    profile_mat = np.asarray(np.vstack([v.toarray() for v in profile_vecs])).squeeze()
                    profile_mean = profile_mat.mean(axis=0, keepdims=True)
                    profile_norm = profile_mean / (np.linalg.norm(profile_mean) + 1e-9)
                    cand = cand_vec.toarray()
                    cand_norm = cand / (np.linalg.norm(cand) + 1e-9)
                    base["sim_tfidf"] = float(np.dot(profile_norm, cand_norm.T).squeeze())
                else:
                    base["sim_tfidf"] = 0.0

            rows.append(base)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col in {"query_id", "item_id"}:
                continue
            df[col] = df[col].fillna(0).astype(np.float32)
        return df


register("similarity", SimilarityFeats)
