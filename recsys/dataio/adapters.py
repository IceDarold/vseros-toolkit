from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import Schema
from .utils_time import filter_by_cutoff

logger = logging.getLogger(__name__)


@dataclass
class AdaptedData:
    interactions: pd.DataFrame
    items: pd.DataFrame
    queries: pd.DataFrame
    schema: Schema

    user_index: Dict[str, int]
    item_index: Dict[str, int]


REQUIRED_INTERACTIONS = {"user_id", "item_id", "ts"}


def _rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    inv = {v: k for k, v in mapping.items() if v in df.columns}
    return df.rename(columns=inv)


def _read_table(path: str) -> pd.DataFrame:
    """Read parquet or CSV depending on extension.

    Parameters
    ----------
    path : str
        File path ending with ``.parquet`` or ``.csv``.
    """

    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format for {path}; use parquet or csv")


def load_datasets(
    *,
    schema: Schema,
    path_interactions: str,
    path_items: Optional[str] = None,
    path_queries: Optional[str] = None,
    cutoff_ts: Optional[pd.Timestamp] = None,
) -> AdaptedData:
    """Load datasets and align column names to the canonical schema.

    Parameters
    ----------
    schema : Schema
        Column mapping configuration.
    path_interactions : str
        Path to interactions parquet/csv.
    cutoff_ts : optional
        If provided, filter interactions to ``ts <= cutoff_ts``.
    """

    interactions = _read_table(path_interactions)
    interactions = _rename_columns(interactions, schema.interactions)

    missing = REQUIRED_INTERACTIONS - set(interactions.columns)
    if missing:
        raise ValueError(f"Missing required interaction columns: {missing}")

    if cutoff_ts is not None:
        interactions = filter_by_cutoff(interactions, cutoff_ts, ts_col="ts")

    if path_items:
        items = _read_table(path_items)
        items = _rename_columns(items, schema.items)
    else:
        items = pd.DataFrame()

    if path_queries:
        queries = _read_table(path_queries)
        queries = _rename_columns(queries, {schema.query_id_col: "query_id"})
        if "query_id" not in queries.columns:
            queries.rename(columns={schema.query_id_col: "query_id"}, inplace=True)
        if "ts" in queries.columns:
            queries["ts"] = pd.to_datetime(queries["ts"])
    else:
        # build queries from interactions based on scope
        if schema.query_scope == "session" and "session_id" in interactions.columns:
            queries = interactions[["session_id"]].drop_duplicates().rename(
                columns={"session_id": "query_id"}
            )
        else:
            queries = interactions[["user_id"]].drop_duplicates().rename(
                columns={"user_id": "query_id"}
            )

    interactions["ts"] = pd.to_datetime(interactions["ts"])
    interactions = interactions.sort_values("ts")

    user_index = {u: i for i, u in enumerate(interactions["user_id"].dropna().unique())}
    item_index = {i: j for j, i in enumerate(interactions["item_id"].dropna().unique())}

    logger.info(
        "Loaded %d interactions, %d items, %d queries",
        len(interactions),
        len(items),
        len(queries),
    )

    return AdaptedData(
        interactions=interactions,
        items=items,
        queries=queries,
        schema=schema,
        user_index=user_index,
        item_index=item_index,
    )
