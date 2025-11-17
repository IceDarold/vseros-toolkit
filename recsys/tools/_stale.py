"""Helpers for stale/up-to-date checks based on fingerprints and mtimes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def read_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_meta(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def is_up_to_date(
    target_files: Iterable[Path],
    deps_files: Iterable[Path],
    *,
    target_fingerprint: str | None,
    deps_fingerprints: Mapping[str, str] | None = None,
    meta_path: Path | None = None,
) -> bool:
    targets = list(target_files)
    deps = list(deps_files)
    if not targets or any(not p.exists() for p in targets):
        return False
    deps_times = [p.stat().st_mtime for p in deps if p.exists()]
    if deps_times:
        newest_dep = max(deps_times)
        if any(p.stat().st_mtime < newest_dep for p in targets):
            return False
    if meta_path and meta_path.exists():
        meta = read_meta(meta_path) or {}
        if target_fingerprint and meta.get("fingerprint") != target_fingerprint:
            return False
        if deps_fingerprints:
            stored = meta.get("deps", {})
            for key, fp in deps_fingerprints.items():
                if stored.get(key) != fp:
                    return False
    return True

