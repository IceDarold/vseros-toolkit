"""Create submission file from recommendations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recsys.tools import _paths
from recsys.tools._base import add_common_args, parse_cli_overrides, prepare_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Create submission CSV")
    add_common_args(parser)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--format", default="kaggle")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out")
    args = parser.parse_args()

    cli_overrides = parse_cli_overrides(args.cli_set)
    prepare_run(subsystem="recsys", dataset_id=args.dataset_id, profile=args.profile, cli_overrides=cli_overrides, verbose=args.verbose)

    recs = pd.read_parquet(args.recommendations)
    recs = recs.sort_values(["query_id", "score_final"], ascending=[True, False]).groupby("query_id").head(args.topk)
    out_dir = Path(args.out) if args.out else _paths.submits_dir(args.dataset_id) / f"{args.tag}.csv"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "kaggle":
        recs[["query_id", "item_id"]].to_csv(out_dir, index=False)
    else:
        recs.to_csv(out_dir, index=False)
    meta = {"rows": len(recs), "tag": args.tag}
    Path(str(out_dir) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
