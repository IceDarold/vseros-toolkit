from __future__ import annotations

import numpy as np


def recall_at_k(pred: list[str], truth: list[str], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(pred[:k]) & set(truth)) / len(set(truth))


def ndcg_at_k(pred: list[str], truth: list[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(pred[:k]):
        if item in truth:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(truth))))
    return dcg / ideal if ideal > 0 else 0.0
