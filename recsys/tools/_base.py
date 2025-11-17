"""Shared helpers for recsys CLI tools.

The utilities here keep individual entrypoints concise by handling config
loading, logging configuration, fingerprint propagation, and resume/skip
decisions. Tools are expected to call :func:`prepare_run` early and use the
returned information to drive their core logic.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from common.configs.loader import ResolvedConfig, load_config


logger = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--profile", help="Profile name or YAML path", required=False)
    parser.add_argument("--set", dest="cli_set", action="append", default=[], help="Override config key=val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")


def parse_cli_overrides(pairs: Iterable[str]) -> Mapping[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            continue
        key, val = raw.split("=", 1)
        cursor: dict[str, Any] = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})  # type: ignore[assignment]
        cursor[parts[-1]] = val
    return overrides


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] [%(levelname)s] %(message)s")


def write_resolved_config(resolved: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        import yaml

        (out_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    except Exception:
        pass


@dataclass
class RunContext:
    cfg: ResolvedConfig
    fingerprint: str


def prepare_run(
    *, subsystem: str, dataset_id: str, profile: str | None, cli_overrides: Mapping[str, Any], verbose: bool
) -> RunContext:
    configure_logging(verbose)
    cfg = load_config(subsystem=subsystem, dataset_id=dataset_id, profile=profile, cli_overrides=cli_overrides)
    return RunContext(cfg=cfg, fingerprint=cfg.fingerprint)

