from __future__ import annotations

import argparse
import logging

from recsys.dataio.adapters import load_datasets
from recsys.dataio.schema import Schema

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Adapt raw data to standard schema")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--items")
    parser.add_argument("--queries")
    args = parser.parse_args()

    schema = Schema.from_yaml(args.schema)
    load_datasets(
        schema=schema,
        path_interactions=args.interactions,
        path_items=args.items,
        path_queries=args.queries,
    )
    logging.info("Data successfully adapted")


if __name__ == "__main__":
    main()
