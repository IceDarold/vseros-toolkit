import hashlib
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _to_bins(values: pd.Series, step_deg: float) -> pd.Series:
    bins = np.floor(values / step_deg)
    bins = pd.Series(bins, index=values.index)
    return bins.fillna(-1).astype(int)


def _meters_to_degrees(step_m: float, lat_ref: float) -> Tuple[float, float]:
    lat_deg = step_m / 111_000
    lon_deg = step_m / (111_000 * max(math.cos(math.radians(lat_ref)), 1e-6))
    return lat_deg, lon_deg


def _select_columns(df: pd.DataFrame, cols: Optional[Sequence[str]]) -> Tuple[str, str]:
    if cols is None:
        raise ValueError("Latitude and longitude columns must be provided explicitly")
    lat_col, lon_col = cols
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("lat_col or lon_col not present in DataFrame")
    return lat_col, lon_col


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    *,
    steps_m: Sequence[int] = (300, 1000),
    prefix: str = "geo",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """Квадратные бины: lat/lon → count и ratio по бинам."""

    lat_col, lon_col = _select_columns(train_df, (lat_col, lon_col))
    if lat_col not in test_df.columns or lon_col not in test_df.columns:
        raise ValueError("lat_col or lon_col not present in test_df")

    params = {"steps_m": list(steps_m), "prefix": prefix, "lat_col": lat_col, "lon_col": lon_col}
    data_stamp = {"train_rows": len(train_df), "test_rows": len(test_df)}
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, code_fingerprint=_fingerprint(), data_stamp=data_stamp)
    if use_cache:
        cached = load_feature_pkg("geo_grid", cache_key)
        if cached is not None:
            return cached

    t0 = time.time()

    lat_ref = pd.concat([train_df[lat_col], test_df[lat_col]]).dropna().mean() if len(train_df) + len(test_df) > 0 else 0.0

    train_features: Dict[str, pd.Series] = {}
    test_features: Dict[str, pd.Series] = {}

    for step in steps_m:
        lat_step, lon_step = _meters_to_degrees(step, lat_ref)

        train_lat_bins = _to_bins(train_df[lat_col], lat_step)
        train_lon_bins = _to_bins(train_df[lon_col], lon_step)
        test_lat_bins = _to_bins(test_df[lat_col], lat_step)
        test_lon_bins = _to_bins(test_df[lon_col], lon_step)

        train_bins = list(zip(train_lat_bins, train_lon_bins))
        test_bins = list(zip(test_lat_bins, test_lon_bins))

        bin_counts = pd.Series(train_bins).value_counts()
        bin_ratios = bin_counts / len(train_df) if len(train_df) > 0 else pd.Series(dtype=float)

        def encode_bins(pairs: List[Tuple[int, int]], table: pd.Series) -> Tuple[pd.Series, pd.Series]:
            counts = []
            ratios = []
            for pair in pairs:
                counts.append(float(table.get(pair, 0.0)))
                ratios.append(float(bin_ratios.get(pair, 0.0)))
            return pd.Series(counts), pd.Series(ratios)

        tr_count, tr_ratio = encode_bins(train_bins, bin_counts)
        te_count, te_ratio = encode_bins(test_bins, bin_counts)

        train_features[f"{prefix}__{step}m__count"] = tr_count.values
        train_features[f"{prefix}__{step}m__ratio"] = tr_ratio.values
        test_features[f"{prefix}__{step}m__count"] = te_count.values
        test_features[f"{prefix}__{step}m__ratio"] = te_ratio.values

    train_out = pd.DataFrame(train_features, index=train_df.index).fillna(0)
    test_out = pd.DataFrame(test_features, index=test_df.index).fillna(0)

    cols = list(train_out.columns)
    meta = {
        "name": "geo_grid",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
    }

    pkg = FeaturePackage(
        name="geo_grid",
        train=train_out,
        test=test_out,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("geo_grid", cache_key, pkg)

    return pkg
