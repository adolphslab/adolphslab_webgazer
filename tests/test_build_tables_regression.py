from __future__ import annotations

from pathlib import Path

import pandas as pd

from alabwebgazer.derive_tables.pipeline import _stable_sort_table, _validate_output_tables


def test_stable_sort_table_orders_df_runs_rows() -> None:
    df = pd.DataFrame(
        [
            {"SID": "S2", "task": "task5", "run_num": 2, "Video": "V2"},
            {"SID": "S1", "task": "task5", "run_num": 3, "Video": "V3"},
            {"SID": "S1", "task": "task5", "run_num": 1, "Video": "V1"},
        ]
    )

    out = _stable_sort_table(df, table_name="df_runs.csv")
    got = out[["SID", "run_num", "Video"]].to_dict("records")
    assert got == [
        {"SID": "S1", "run_num": 1, "Video": "V1"},
        {"SID": "S1", "run_num": 3, "Video": "V3"},
        {"SID": "S2", "run_num": 2, "Video": "V2"},
    ]


def test_validate_output_tables_reports_missing_columns(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)

    # Missing required feature-table columns by construction.
    pd.DataFrame([{"SID": "S1", "Video": "V1"}]).to_csv(
        tables_dir / "df_mergedmain.csv",
        index=False,
    )
    pd.DataFrame([{"SID": "S1", "Video": "V1", "Group": "Control"}]).to_csv(
        tables_dir / "df_runs.csv",
        index=False,
    )

    report = _validate_output_tables(
        tables_dir=tables_dir,
        allowed_groups=("Control",),
        required_features=("speaker",),
    )

    assert report["errors"]
    assert any("missing required columns" in msg for msg in report["errors"])


def test_legacy_algorithm_modules_are_single_source() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "alabwebgazer"

    dwell_candidates = sorted(root.rglob("compute_dwelltime.py"))
    cluster_candidates = sorted(root.rglob("clustering_dataprep.py"))

    assert dwell_candidates == [root / "legacy" / "compute_dwelltime.py"]
    assert cluster_candidates == [root / "legacy" / "clustering_dataprep.py"]
