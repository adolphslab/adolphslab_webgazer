from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alabwebgazer.analysis_ready.config import ConfigError as AnalysisConfigError
from alabwebgazer.analysis_ready.config import load_config as load_analysis_config
from alabwebgazer.derive_tables.config import ConfigError as BuildTablesConfigError
from alabwebgazer.derive_tables.config import load_config as load_build_tables_config
from alabwebgazer.preprocessing.config import ConfigError as PreprocessConfigError
from alabwebgazer.preprocessing.config import load_config as load_preprocess_config


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_preprocess_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    cfg = {
        "paths": {"input_glob": "examples/raw_demo/*.json"},
        "condition_rules": [{"pattern": "demo", "condition": "demo", "video_length_seconds": 5.0}],
        "unknown_key": 123,
    }
    cfg_path = tmp_path / "pre.yml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(PreprocessConfigError):
        _ = load_preprocess_config(cfg_path)


def test_build_tables_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    cfg = {
        "dwell_tasks": [
            {
                "task": "task5_pro",
                "window_data_dir": "windows",
                "video_feats_dir": "video_feats",
            }
        ],
        "unknown_key": 123,
    }
    cfg_path = tmp_path / "build.yml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(BuildTablesConfigError):
        _ = load_build_tables_config(cfg_path)


def test_analysis_ready_config_rejects_unknown_top_level_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = {
        "run": {"run_id": "r1", "sap_name": "SAP", "sap_version": "TEST"},
        "paths": {"feature_long_csv": "feat.csv", "run_level_csv": "runs.csv"},
        "filters": {
            "tasks_keep": ["task5_pro"],
            "groups_keep": ["Control", "ASD"],
            "sex_keep": ["Female", "Male"],
            "design_videos": ["S1"],
        },
        "confirmatory": {
            "contrasts": [["Control", "ASD"]],
            "endpoints": {
                "H3_switching_overall": {
                    "table": "run_level",
                    "successes_col": "n_switches",
                    "trials_col": "n_contig_steps",
                    "predictors": ["Group", "Video", "Sex"],
                }
            },
        },
        "unknown_key": 123,
    }
    cfg_path = tmp_path / "analysis.yml"
    _write_yaml(cfg_path, cfg)
    monkeypatch.setenv("ALABWEBGAZER_SKIP_INPUT_EXISTS_CHECK", "1")

    with pytest.raises(AnalysisConfigError):
        _ = load_analysis_config(cfg_path)
