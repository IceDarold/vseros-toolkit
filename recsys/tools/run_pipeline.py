"""Orchestrate recsys stages in a single command.

The pipeline is intentionally lightweight: it chains individual CLI tools and
skips work when artifacts are already up to date (fingerprint hit).
"""
from __future__ import annotations

import argparse
import subprocess
from typing import Iterable

from recsys.tools._base import add_common_args, parse_cli_overrides, prepare_run


STAGE_TO_CMD = {
    "adapt": "recsys/tools/run_adapt.py",
    "sessionize": "recsys/tools/run_sessionize.py",
    "splits": "recsys/tools/run_splits.py",
    "build_pairs": "recsys/tools/run_build_pairs.py",
    "candidates": "recsys/tools/run_candidates.py",
    "features": "recsys/tools/run_features.py",
    "ranker": "recsys/tools/run_ranker.py",
    "blend": "recsys/tools/run_blend.py",
    "rerank": "recsys/tools/run_rerank.py",
    "eval": "recsys/tools/run_eval.py",
    "submit": "recsys/tools/run_submit.py",
}


def _run_stage(stage: str, args: argparse.Namespace, extra: list[str]) -> None:
    cmd = ["python", STAGE_TO_CMD[stage], "--dataset_id", args.dataset_id]
    if args.profile:
        cmd += ["--profile", args.profile]
    if stage == "adapt":
        cmd += ["--schema", args.schema, "--interactions", args.interactions]
        if args.items:
            cmd += ["--items", args.items]
        if args.out_dir:
            cmd += ["--out_dir", args.out_dir]
    elif stage == "splits":
        cmd += ["--schema", args.schema, "--interactions", args.interactions]
        if args.splits_out:
            cmd += ["--out_path", args.splits_out]
    elif stage == "build_pairs":
        cmd += ["--schema", args.schema, "--interactions", args.interactions]
        if args.queries:
            cmd += ["--queries", args.queries]
        if args.pairs_out:
            cmd += ["--out_path", args.pairs_out]
    cmd += extra
    subprocess.check_call(cmd)


def parse_stage_list(raw: str) -> Iterable[str]:
    return [p for p in raw.split(",") if p]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recsys pipeline")
    add_common_args(parser)
    parser.add_argument("--stages", default="adapt,splits,build_pairs")
    parser.add_argument("--until")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--items")
    parser.add_argument("--queries")
    parser.add_argument("--out_dir")
    parser.add_argument("--splits_out")
    parser.add_argument("--pairs_out")
    args = parser.parse_args()

    cli_overrides = parse_cli_overrides(args.cli_set)
    prepare_run(subsystem="recsys", dataset_id=args.dataset_id, profile=args.profile, cli_overrides=cli_overrides, verbose=args.verbose)

    stages = list(parse_stage_list(args.stages))
    if args.until and args.until in stages:
        stages = stages[: stages.index(args.until) + 1]
    for stage in stages:
        _run_stage(stage, args, [])


if __name__ == "__main__":  # pragma: no cover
    main()
