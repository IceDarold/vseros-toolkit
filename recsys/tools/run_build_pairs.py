"""CLI: build train/val/test pairs with negative sampling."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recsys.dataio.adapters import load_interactions, load_queries
from recsys.dataio.pairs import build_pairs
from recsys.dataio.queries import build_queries
from recsys.dataio.schema import Schema
from recsys.tools import _paths, _stale
from recsys.tools._base import add_common_args, parse_cli_overrides, prepare_run, write_resolved_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairs for ranking")
    add_common_args(parser)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--queries")
    parser.add_argument("--scope", default="session")
    parser.add_argument("--neg_pos_ratio", type=int, default=10)
    parser.add_argument("--neg_strategy", default="pop")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    cli_overrides = parse_cli_overrides(args.cli_set)
    ctx = prepare_run(
        subsystem="recsys", dataset_id=args.dataset_id, profile=args.profile, cli_overrides=cli_overrides, verbose=args.verbose
    )
    out_path = Path(args.out_path) if args.out_path else _paths.dataio_dir(args.dataset_id) / "pairs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = f"pairs-{ctx.fingerprint}-{args.scope}-{args.neg_pos_ratio}-{args.neg_strategy}"
    meta_path = out_path.with_suffix(".meta.json")
    deps = [Path(args.interactions)]
    if args.resume and _stale.is_up_to_date([out_path], deps, target_fingerprint=fingerprint, meta_path=meta_path):
        print("Up-to-date, skipping build_pairs")
        return
    if args.dry_run:
        print("Dry-run: would write pairs to", out_path)
        return

    t0 = time.time()
    schema = Schema.from_yaml(args.schema)
    inter = load_interactions(args.interactions, schema)
    queries_df = load_queries(args.queries, schema)
    if queries_df.empty:
        queries_df = build_queries(inter, scope=args.scope)
    pairs = build_pairs(
        inter,
        queries_df,
        neg_pos_ratio=args.neg_pos_ratio,
        neg_strategy=args.neg_strategy,
        rng=np.random.RandomState(args.seed),
        scope=args.scope,
    )
    pairs.to_parquet(out_path, index=False)
    report = {
        "pairs": len(pairs),
        "queries": pairs["query_id"].nunique() if "query_id" in pairs.columns else 0,
        "fingerprint": fingerprint,
        "duration_sec": time.time() - t0,
    }
    (out_path.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_resolved_config(ctx.cfg.resolved, out_path.parent)
    _stale.write_meta(meta_path, {"fingerprint": fingerprint})


if __name__ == "__main__":  # pragma: no cover
    main()
