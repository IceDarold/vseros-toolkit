from __future__ import annotations

import dataclasses
from typing import Any, Dict

import yaml


@dataclasses.dataclass
class Schema:
    """Container for column mapping and query scope.

    Attributes
    ----------
    interactions: Dict[str, str]
        Mapping from canonical interaction field name to dataset column name.
    items: Dict[str, str]
        Mapping from canonical item field name to dataset column name.
    queries: Dict[str, Any]
        Query settings with ``scope`` and ``id_col``.
    """

    interactions: Dict[str, str]
    items: Dict[str, str]
    queries: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "Schema":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            interactions=cfg.get("interactions", {}),
            items=cfg.get("items", {}),
            queries=cfg.get("queries", {}),
        )

    @property
    def query_id_col(self) -> str:
        return self.queries.get("id_col", "query_id")

    @property
    def query_scope(self) -> str:
        return self.queries.get("scope", "session")

    def canonical_to_source(self, section: str, name: str) -> str:
        mapping = getattr(self, section)
        return mapping.get(name, name)

    def source_to_canonical(self, section: str, name: str) -> str:
        mapping = getattr(self, section)
        inv = {v: k for k, v in mapping.items()}
        return inv.get(name, name)
