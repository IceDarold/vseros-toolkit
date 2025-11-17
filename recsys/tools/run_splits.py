"""CLI: assign time-based splits with embargo.

Example:
    python recsys/tools/run_splits.py --dataset_id demo --schema recsys/configs/schema.yaml \
        --interactions data/interactions.csv --train_until 2024-01-01 \
        --val_until 2024-02-01 --out_path artifacts/recsys/dataio/demo/interactions_splits.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recsys.dataio.adapters import load_interactions
from recsys.dataio.schema import Schema
from recsys.dataio.splits import SplitConfig, assign_time_splits, save_split_report
from recsys.tools import _paths, _stale
from recsys.tools._base import add_common_args, parse_cli_overrides, prepare_run, write_resolved_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign time splits")
    add_common_args(parser)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--train_until")
    parser.add_argument("--val_until")
    parser.add_argument("--embargo", default="0D")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    cli_overrides = parse_cli_overrides(args.cli_set)
    ctx = prepare_run(
        subsystem="recsys", dataset_id=args.dataset_id, profile=args.profile, cli_overrides=cli_overrides, verbose=args.verbose
    )
    out_path = Path(args.out_path) if args.out_path else _paths.dataio_dir(args.dataset_id) / "interactions_splits.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = f"splits-{ctx.fingerprint}"
    meta_path = out_path.with_suffix(".meta.json")
    if args.resume and _stale.is_up_to_date([out_path], [Path(args.interactions)], target_fingerprint=fingerprint, meta_path=meta_path):
        print("Up-to-date, skipping splits")
        return
    if args.dry_run:
        print("Dry-run: would write splits to", out_path)
        return

    t0 = time.time()
    schema = Schema.from_yaml(args.schema)
    inter = load_interactions(args.interactions, schema)
    cfg = SplitConfig(train_until=args.train_until, val_until=args.val_until, embargo=args.embargo)
    inter_split = assign_time_splits(inter, cfg)
    inter_split.to_parquet(out_path, index=False)
    save_split_report(inter_split, str(out_path.with_suffix(".json")))
    write_resolved_config(ctx.cfg.resolved, out_path.parent)
    _stale.write_meta(meta_path, {"fingerprint": fingerprint, "duration_sec": time.time() - t0})


if __name__ == "__main__":  # pragma: no cover
    main()
