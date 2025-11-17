from __future__ import annotations

import subprocess
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_adapt_and_pairs_smoke(tmp_path: Path) -> None:
    dataset = "tiny_smoke"
    inter = FIXTURES / "tiny_interactions.csv"
    items = FIXTURES / "tiny_items.csv"
    schema = Path("recsys/configs/schema.yaml")
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapt_out = out_dir / "adapt"
    subprocess.check_call(
        [
            "python",
            "-m",
            "recsys.tools.run_adapt",
            "--dataset_id",
            dataset,
            "--schema",
            str(schema),
            "--interactions",
            str(inter),
            "--items",
            str(items),
            "--out_dir",
            str(adapt_out),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    assert (adapt_out / "interactions_norm.parquet").exists()

    pairs_out = out_dir / "pairs.parquet"
    subprocess.check_call(
        [
            "python",
            "-m",
            "recsys.tools.run_build_pairs",
            "--dataset_id",
            dataset,
            "--schema",
            str(schema),
            "--interactions",
            str(adapt_out / "interactions_norm.parquet"),
            "--queries",
            str(FIXTURES / "tiny_queries.csv"),
            "--out_path",
            str(pairs_out),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    assert pairs_out.exists()


def test_resume_skip(tmp_path: Path) -> None:
    dataset = "tiny_resume"
    inter = FIXTURES / "tiny_interactions.csv"
    schema = Path("recsys/configs/schema.yaml")
    out_dir = tmp_path / "adapt"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "recsys.tools.run_adapt",
        "--dataset_id",
        dataset,
        "--schema",
        str(schema),
        "--interactions",
        str(inter),
        "--out_dir",
        str(out_dir),
    ]
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[2])
    target = out_dir / "interactions_norm.parquet"
    mtime = target.stat().st_mtime
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[2])
    assert target.stat().st_mtime == mtime


def test_pipeline_subset(tmp_path: Path) -> None:
    dataset = "tiny_pipe"
    inter = FIXTURES / "tiny_interactions.csv"
    schema = Path("recsys/configs/schema.yaml")
    out_base = tmp_path / "artifacts"
    out_base.mkdir(parents=True, exist_ok=True)

    # run pipeline stages using custom outputs to stay within tmp_path
    subprocess.check_call(
        [
        "python",
        "-m",
        "recsys.tools.run_adapt",
            "--dataset_id",
            dataset,
            "--schema",
            str(schema),
            "--interactions",
            str(inter),
            "--out_dir",
            str(out_base / "dataio"),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    subprocess.check_call(
        [
        "python",
        "-m",
        "recsys.tools.run_splits",
            "--dataset_id",
            dataset,
            "--schema",
            str(schema),
            "--interactions",
            str(out_base / "dataio" / "interactions_norm.parquet"),
            "--out_path",
            str(out_base / "dataio" / "interactions_splits.parquet"),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    assert (out_base / "dataio" / "interactions_splits.parquet").exists()

