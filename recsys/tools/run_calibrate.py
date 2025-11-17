"""Optional probability calibration helper."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from recsys.rankers.calibration import platt_scale
from recsys.tools._base import add_common_args, parse_cli_overrides, prepare_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ranker scores")
    add_common_args(parser)
    parser.add_argument("--oof_true", required=True)
    parser.add_argument("--oof_pred", required=True)
    parser.add_argument("--method", default="platt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cli_overrides = parse_cli_overrides(args.cli_set)
    prepare_run(subsystem="recsys", dataset_id=args.dataset_id, profile=args.profile, cli_overrides=cli_overrides, verbose=args.verbose)

    if args.dry_run:
        print("Dry-run: calibration")
        return
    y_true = np.load(args.oof_true)
    y_pred = np.load(args.oof_pred)
    if args.method == "platt":
        calibrated = platt_scale(y_true, y_pred)
    else:  # pragma: no cover - placeholder
        calibrated = y_pred
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, calibrated)


if __name__ == "__main__":  # pragma: no cover
    main()
