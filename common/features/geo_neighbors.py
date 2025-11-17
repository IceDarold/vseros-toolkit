import hashlib
import math
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _meters_to_radians(distance_m: float) -> float:
    return distance_m / 6_371_000.0


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    *,
    radii_m: Sequence[int] = (300, 1000),
    prefix: str = "geo_nb",
    use_cache: bool = True,
    cache_key_extra: Optional[Dict] = None,
) -> FeaturePackage:
    """BallTree(haversine): число соседей и плотности в радиусах."""

    params = {
        "radii_m": list(radii_m),
        "prefix": prefix,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }
    data_stamp = {"train_rows": len(train_df), "test_rows": len(test_df)}
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, code_fingerprint=_fingerprint(), data_stamp=data_stamp)
    if use_cache:
        cached = load_feature_pkg("geo_neighbors", cache_key)
        if cached is not None:
            return cached

    try:
        from sklearn.neighbors import BallTree
    except Exception:
        print("[geo_neighbors] sklearn not available, returning empty package")
        empty_train = pd.DataFrame(index=train_df.index)
        empty_test = pd.DataFrame(index=test_df.index)
        pkg = FeaturePackage(
            name="geo_neighbors",
            train=empty_train,
            test=empty_test,
            kind="dense",
            cols=[],
            meta={
                "name": "geo_neighbors",
                "params": params,
                "time_sec": 0.0,
                "cache_key": cache_key,
                "deps": [],
            },
        )
        if use_cache:
            save_feature_pkg("geo_neighbors", cache_key, pkg)
        return pkg

    if lat_col not in train_df.columns or lon_col not in train_df.columns:
        raise ValueError("lat_col or lon_col not present in train_df")
    if lat_col not in test_df.columns or lon_col not in test_df.columns:
        raise ValueError("lat_col or lon_col not present in test_df")

    t0 = time.time()

    coords_train = np.radians(np.column_stack((train_df[lat_col].values, train_df[lon_col].values)))
    coords_test = np.radians(np.column_stack((test_df[lat_col].values, test_df[lon_col].values)))

    coords_train = np.nan_to_num(coords_train, nan=0.0)
    coords_test = np.nan_to_num(coords_test, nan=0.0)

    tree = BallTree(coords_train, metric="haversine")

    train_features: Dict[str, np.ndarray] = {}
    test_features: Dict[str, np.ndarray] = {}

    for radius in radii_m:
        radius_rad = _meters_to_radians(radius)
        counts_train = tree.query_radius(coords_train, r=radius_rad, count_only=True)
        counts_test = tree.query_radius(coords_test, r=radius_rad, count_only=True)

        area = math.pi * (radius ** 2)
        density_train = counts_train / area if area > 0 else counts_train
        density_test = counts_test / area if area > 0 else counts_test

        train_features[f"{prefix}__r{radius}m__count"] = counts_train
        train_features[f"{prefix}__r{radius}m__density"] = density_train
        test_features[f"{prefix}__r{radius}m__count"] = counts_test
        test_features[f"{prefix}__r{radius}m__density"] = density_test

    train_out = pd.DataFrame(train_features, index=train_df.index)
    test_out = pd.DataFrame(test_features, index=test_df.index)

    cols = list(train_out.columns)
    meta = {
        "name": "geo_neighbors",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": [],
    }

    pkg = FeaturePackage(
        name="geo_neighbors",
        train=train_out,
        test=test_out,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("geo_neighbors", cache_key, pkg)

    return pkg
