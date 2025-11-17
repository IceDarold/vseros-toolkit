"""Helper utilities for recsys tool paths.

This module centralizes artifact locations so individual CLI tools can remain
small and only care about their functional logic. All paths are relative to
the repository root and follow the ``artifacts/recsys`` layout used across
dataio, candidates, features, models, rerank, and eval stages.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART_ROOT = ROOT / "artifacts" / "recsys"


def dataio_dir(dataset_id: str) -> Path:
    return ART_ROOT / "dataio" / dataset_id


def candidates_dir(dataset_id: str, profile: str) -> Path:
    return ART_ROOT / "candidates" / dataset_id / profile


def features_dir(dataset_id: str, profile: str) -> Path:
    return ART_ROOT / "features" / dataset_id / profile


def models_dir(dataset_id: str) -> Path:
    return ART_ROOT / "models" / dataset_id


def rerank_dir(dataset_id: str, profile: str) -> Path:
    return ART_ROOT / "rerank" / dataset_id / profile


def eval_dir(dataset_id: str, profile: str) -> Path:
    return ART_ROOT / "eval" / dataset_id / profile


def submits_dir(dataset_id: str) -> Path:
    return ART_ROOT / "submits" / dataset_id

