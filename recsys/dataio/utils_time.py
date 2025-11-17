from __future__ import annotations

import pandas as pd


def filter_by_cutoff(df: pd.DataFrame, cutoff_ts, ts_col: str = "ts") -> pd.DataFrame:
    cutoff_ts = pd.to_datetime(cutoff_ts)
    return df[df[ts_col] <= cutoff_ts].copy()


def assert_time_safe(candidates: pd.DataFrame, queries: pd.DataFrame, *, ts_col: str = "ts") -> bool:
    if ts_col not in queries.columns:
        return True
    merged = candidates.merge(queries[["query_id", ts_col]], on="query_id", how="left")
    if merged[ts_col].isna().all():
        return True
    unsafe = merged[merged[ts_col + "_x"] > merged[ts_col + "_y"]] if ts_col + "_x" in merged.columns else merged[merged[ts_col] > merged[ts_col + "_y"]]
    if not unsafe.empty:
        raise AssertionError("Found candidates with timestamp after query ts")
    return True
