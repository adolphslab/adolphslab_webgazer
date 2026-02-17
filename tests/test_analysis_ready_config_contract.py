from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from alabwebgazer.analysis_ready.config import ConfigError, load_config


def _write_min_inputs(tmp_path: Path) -> tuple[Path, Path]:
    feat = tmp_path / "df_mergedmain.csv"
    runs = tmp_path / "df_runs.csv"

    pd.DataFrame(
        [
            {
                "SID": "S1",
                "Video": "S1",
                "task": "task5_pro",
                "feature": "speaker",
                "Group": "Control",
                "Sex": "Female",
                "dwell_bins": 1,
                "n_present_any_valid": 2,
            }
        ]
    ).to_csv(feat, index=False)

    pd.DataFrame(
        [
            {
                "SID": "S1",
                "Video": "S1",
                "task": "task5_pro",
                "Group": "Control",
                "Sex": "Female",
                "n_switches": 0,
                "n_contig_steps": 2,
            }
        ]
    ).to_csv(runs, index=False)

    return feat, runs


def _base_cfg(tmp_path: Path, feat: Path, runs: Path) -> dict:
    return {
        "run": {
            "run_id": "cfg_contract_test",
            "random_seed": 1,
            "sap_name": "SAP_implemented_all",
            "sap_version": "TEST",
            "notes": "",
        },
        "paths": {
            "feature_long_csv": str(feat),
            "run_level_csv": str(runs),
            "output_root": str(tmp_path / "runs"),
        },
        "filters": {
            "tasks_keep": ["task5_pro"],
            "use_run_only": True,
            "groups_keep": ["Control", "ASD"],
            "sex_keep": ["Female", "Male"],
            "design_videos": ["S1", "S2"],
        },
        "confirmatory": {
            "contrasts": [["Control", "ASD"]],
            "endpoints": {
                "H1_speaker": {
                    "table": "feature_long",
                    "feature_filter": ["speaker"],
                    "successes_col": "dwell_bins",
                    "trials_col": "n_present_any_valid",
                    "predictors": ["Group", "Video", "Sex"],
                }
            },
        },
        "models": {
            "primary_backend": "glm_cluster",
            "cluster_col": "SID",
            "bootstrap": {"enabled": False, "n_resamples": 200, "seed": 1},
        },
        "multiplicity": {"method": "bh_fdr"},
    }


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


def test_config_requires_explicit_filter_lists(tmp_path: Path) -> None:
    feat, runs = _write_min_inputs(tmp_path)
    cfg = _base_cfg(tmp_path, feat, runs)

    del cfg["filters"]["tasks_keep"]
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ConfigError):
        _ = load_config(cfg_path)


def test_config_requires_filters_block(tmp_path: Path) -> None:
    feat, runs = _write_min_inputs(tmp_path)
    cfg = _base_cfg(tmp_path, feat, runs)

    del cfg["filters"]
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ConfigError):
        _ = load_config(cfg_path)


def test_config_rejects_empty_groups_keep(tmp_path: Path) -> None:
    feat, runs = _write_min_inputs(tmp_path)
    cfg = _base_cfg(tmp_path, feat, runs)

    cfg["filters"]["groups_keep"] = []
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ConfigError):
        _ = load_config(cfg_path)


def test_feature_long_endpoint_requires_explicit_feature_filter(tmp_path: Path) -> None:
    feat, runs = _write_min_inputs(tmp_path)
    cfg = _base_cfg(tmp_path, feat, runs)

    del cfg["confirmatory"]["endpoints"]["H1_speaker"]["feature_filter"]
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ConfigError):
        _ = load_config(cfg_path)


def test_contrasts_must_be_subset_of_groups_keep(tmp_path: Path) -> None:
    feat, runs = _write_min_inputs(tmp_path)
    cfg = _base_cfg(tmp_path, feat, runs)

    cfg["confirmatory"]["contrasts"] = [["Control", "SPARK"]]
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ConfigError):
        _ = load_config(cfg_path)


def test_analysis_ready_paths_resolve_relative_to_config_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    cfg_dir = tmp_path / "cfgs"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    feat = data_dir / "df_mergedmain.csv"
    runs = data_dir / "df_runs.csv"
    pd.DataFrame(
        [
            {
                "SID": "S1",
                "Video": "S1",
                "task": "task5_pro",
                "feature": "speaker",
                "Group": "Control",
                "Sex": "Female",
                "dwell_bins": 1,
                "n_present_any_valid": 2,
            }
        ]
    ).to_csv(feat, index=False)
    pd.DataFrame(
        [
            {
                "SID": "S1",
                "Video": "S1",
                "task": "task5_pro",
                "Group": "Control",
                "Sex": "Female",
                "n_switches": 0,
                "n_contig_steps": 2,
            }
        ]
    ).to_csv(runs, index=False)

    cfg = _base_cfg(tmp_path, feat, runs)
    cfg["paths"] = {
        "feature_long_csv": "../data/df_mergedmain.csv",
        "run_level_csv": "../data/df_runs.csv",
        "output_root": "../runs_analysis_ready",
    }
    cfg_path = cfg_dir / "analysis_ready.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loaded = load_config(cfg_path)
    assert loaded.paths.feature_long_csv == (cfg_dir / "../data/df_mergedmain.csv").resolve()
    assert loaded.paths.run_level_csv == (cfg_dir / "../data/df_runs.csv").resolve()
    assert loaded.paths.output_root == (cfg_dir / "../runs_analysis_ready").resolve()
