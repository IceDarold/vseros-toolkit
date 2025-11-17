"""CLI: adapt raw interactions/items to unified schema and optionally save indexers.

Example:
    python recsys/tools/run_adapt.py --dataset_id demo --schema recsys/configs/schema.yaml \
        --interactions data/interactions.csv --items data/items.csv \
        --out_dir artifacts/recsys/dataio/demo --save_indexers 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema
from recsys.tools import _paths, _stale
from recsys.tools._base import RunContext, add_common_args, parse_cli_overrides, prepare_run, write_resolved_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt raw data to standard schema")
    add_common_args(parser)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--items")
    parser.add_argument("--queries")
    parser.add_argument("--out_dir")
    parser.add_argument("--save_indexers", type=int, default=1)
    args = parser.parse_args()

    cli_overrides = parse_cli_overrides(args.cli_set)
    ctx: RunContext = prepare_run(
        subsystem="recsys", dataset_id=args.dataset_id, profile=args.profile, cli_overrides=cli_overrides, verbose=args.verbose
    )

    out_dir = Path(args.out_dir) if args.out_dir else _paths.dataio_dir(args.dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_schema = Schema.from_yaml(args.schema)
    fingerprint = f"adapt-{ctx.fingerprint}"
    targets = [out_dir / "interactions_norm.parquet"]
    meta_path = out_dir / "meta_adapt.json"
    if args.resume and _stale.is_up_to_date(targets, [Path(args.interactions)], target_fingerprint=fingerprint, meta_path=meta_path):
        print("Up-to-date, skipping adapt")
        return
    if args.dry_run:
        print("Dry-run: would adapt data to", out_dir)
        return

    t0 = time.time()
    data = load_datasets(
        schema=resolved_schema,
        path_interactions=args.interactions,
        path_items=args.items,
        path_queries=args.queries,
        save_indexers=str(out_dir) if args.save_indexers else None,
    )
    data.interactions.to_parquet(out_dir / "interactions_norm.parquet")
    if not data.items.empty:
        data.items.to_parquet(out_dir / "items_norm.parquet")
    if not data.queries.empty:
        data.queries.to_parquet(out_dir / "queries_norm.parquet")

    report = {
        "counts": {
            "interactions": len(data.interactions),
            "items": len(data.items),
            "queries": len(data.queries),
        },
        "fingerprint": fingerprint,
        "duration_sec": time.time() - t0,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_resolved_config(ctx.cfg.resolved, out_dir)
    _stale.write_meta(meta_path, {"fingerprint": fingerprint})


if __name__ == "__main__":  # pragma: no cover
    main()
