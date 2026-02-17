from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from alabwebgazer.core.config import load_config
from alabwebgazer.core.pipeline import PipelineError, run_pipeline


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_unified_pipeline_smoke_with_stage_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "subj_demo.json").write_text("[]", encoding="utf-8")

    demo_dir = tmp_path / "demo"
    (demo_dir / "video_feats").mkdir(parents=True, exist_ok=True)
    (demo_dir / "runlists").mkdir(parents=True, exist_ok=True)
    (demo_dir / "window_data").mkdir(parents=True, exist_ok=True)
    (demo_dir / "runlists" / "runlist_task5_pro.csv").write_text("run_num\n1\n", encoding="utf-8")

    pre_cfg = tmp_path / "pre.yml"
    _write_yaml(
        pre_cfg,
        {
            "run": {"run_id": None, "random_seed": 0, "overwrite": False},
            "paths": {
                "input_glob": "raw/*.json",
                "output_root": "runs_preprocess",
                "windows_subdir": "windows",
                "summary_csv": "preprocess_summary.csv",
            },
            "filename_parsing": {
                "delimiter": "_",
                "id_index": 0,
                "session_start": 1,
                "session_end": 2,
            },
            "trials": {"trial_type": "video-keyboard-response", "require_webgazer_data": True},
            "condition_rules": [
                {
                    "pattern": "demo_video",
                    "condition": "demo",
                    "video_length_seconds": 5.0,
                    "match_field": "stimname",
                    "case_sensitive": False,
                    "use_regex": False,
                }
            ],
            "preprocess": {
                "median_kernel": 3,
                "median_method": "scipy_medfilt",
                "drift_mode": "none",
                "quadrant_method": "midpoint",
                "allow_midpoint_fallback": False,
                "x_prob_threshold": 0.9,
                "y_prob_threshold": 0.7,
                "random_state": 0,
                "bin_seconds": 0.5,
                "resample_edge_policy": "half_open",
                "resample_include_endpoint": False,
            },
            "output_naming": {"template": "videoview_window_{id}_{session}_{condition}.csv"},
            "privacy": {"hash_subject_ids": False, "hash_salt_env": "ALABWEBGAZER_HASH_SALT"},
        },
    )

    bt_cfg = tmp_path / "bt.yml"
    _write_yaml(
        bt_cfg,
        {
            "run": {"run_id": None, "random_seed": 0, "overwrite": False},
            "paths": {"output_root": "runs_tables"},
            "dwell_tasks": [
                {
                    "task": "task5_pro",
                    "window_data_dir": "demo/window_data",
                    "video_feats_dir": "demo/video_feats",
                    "window_glob": "videoview*.csv",
                    "runlist_csv": "demo/runlists/runlist_task5_pro.csv",
                }
            ],
            "cluster": {"enabled": True, "pro_task": "task5_pro", "spark_task": None},
        },
    )

    ar_cfg = tmp_path / "ar.yml"
    _write_yaml(
        ar_cfg,
        {
            "run": {
                "run_id": "analysis_ready",
                "random_seed": 0,
                "sap_name": "SAP_implemented_all",
                "sap_version": "TEST",
            },
            "paths": {
                "feature_long_csv": "demo/feature.csv",
                "run_level_csv": "demo/runs.csv",
                "output_root": "runs",
            },
            "filters": {
                "tasks_keep": ["task5_pro"],
                "use_run_only": True,
                "groups_keep": ["Control", "ASD"],
                "sex_keep": ["Female", "Male"],
                "design_videos": ["S1"],
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
        },
    )

    unified_cfg_path = tmp_path / "full.yml"
    _write_yaml(
        unified_cfg_path,
        {
            "run": {
                "output_root": str(tmp_path / "runs_pipeline"),
                "run_id": "u1",
                "random_seed": 0,
                "overwrite": False,
            },
            "pipeline": {
                "steps": ["preprocess_jspsych", "build_tables", "analysis_ready"],
            },
            "preprocess_jspsych": {"enabled": True, "config_path": str(pre_cfg)},
            "build_tables": {
                "enabled": True,
                "config_path": str(bt_cfg),
                "use_preprocess_windows": True,
            },
            "analysis_ready": {
                "enabled": True,
                "config_path": str(ar_cfg),
                "use_build_tables_outputs": True,
                "steps": ["ingest_analysis_ready"],
            },
        },
    )

    def fake_preprocess(*, config_path: Path) -> Path:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_dir = Path(cfg["paths"]["output_root"]) / cfg["run"]["run_id"]
        (run_dir / "windows").mkdir(parents=True, exist_ok=True)
        (run_dir / "windows" / "videoview_window_demo.csv").write_text(
            "Tstart,Tend,window\n0.0,0.5,1\n",
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(
            json.dumps({"requirements": ["RAW-WINDOW-001"]}),
            encoding="utf-8",
        )
        return run_dir

    def fake_build_tables(*, config_path: Path) -> Path:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_dir = Path(cfg["paths"]["output_root"]) / cfg["run"]["run_id"]
        window_dir = Path(cfg["dwell_tasks"][0]["window_data_dir"])
        assert window_dir.as_posix().endswith("stages/01_preprocess_windows/windows")

        tables = run_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        (tables / "df_mergedmain.csv").write_text(
            "SID,Video,task,feature,Group,Sex,dwell_bins,n_present_any_valid\n"
            "S1,S1,task5_pro,speaker,Control,Female,1,2\n",
            encoding="utf-8",
        )
        (tables / "df_runs.csv").write_text(
            "SID,Video,task,Group,Sex,n_switches,n_contig_steps\n"
            "S1,S1,task5_pro,Control,Female,0,2\n",
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(
            json.dumps({"requirements": ["WINDOW-TABLES-001", "WINDOW-TABLES-002"]}),
            encoding="utf-8",
        )
        return run_dir

    def fake_analysis_ready(*, config_path: Path, steps: list[str] | None, overwrite: bool) -> Path:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["paths"]["feature_long_csv"].endswith(
            "stages/02_build_tables/tables/df_mergedmain.csv"
        )
        assert cfg["paths"]["run_level_csv"].endswith("stages/02_build_tables/tables/df_runs.csv")
        assert steps == ["ingest_analysis_ready"]
        assert overwrite is False

        run_dir = Path(cfg["paths"]["output_root"]) / cfg["run"]["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"status": "completed", "requirements_used": ["SAP-IMPL-INGEST-001"]}),
            encoding="utf-8",
        )
        return run_dir

    import alabwebgazer.analysis_ready.pipeline as ar_pipeline
    import alabwebgazer.derive_tables.pipeline as bt_pipeline
    import alabwebgazer.preprocessing.batch as pre_batch

    monkeypatch.setattr(pre_batch, "run_preprocess_pipeline", fake_preprocess)
    monkeypatch.setattr(bt_pipeline, "run_pipeline", fake_build_tables)
    monkeypatch.setattr(ar_pipeline, "run_pipeline", fake_analysis_ready)

    cfg = load_config(unified_cfg_path)
    run_dir = run_pipeline(config_path=unified_cfg_path, cfg=cfg)

    assert run_dir == tmp_path / "runs_pipeline" / "u1"
    assert (run_dir / "stage_configs" / "preprocess_jspsych.yml").exists()
    assert (run_dir / "stage_configs" / "build_tables.yml").exists()
    assert (run_dir / "stage_configs" / "analysis_ready.yml").exists()
    assert (run_dir / "stages" / "01_preprocess_windows").exists()
    assert (run_dir / "stages" / "02_build_tables").exists()
    assert (run_dir / "stages" / "03_analysis_ready").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["steps_requested"] == ["preprocess_jspsych", "build_tables", "analysis_ready"]
    assert manifest["stage_run_dirs"]["preprocess_jspsych"] == "stages/01_preprocess_windows"
    assert "RAW-WINDOW-001" in manifest["requirements"]
    assert "WINDOW-TABLES-002" in manifest["requirements"]
    assert "SAP-IMPL-INGEST-001" in manifest["requirements"]


def test_unified_pipeline_enforces_preprocess_before_build_when_wired(tmp_path: Path) -> None:
    bt_cfg = tmp_path / "bt.yml"
    _write_yaml(
        bt_cfg,
        {
            "run": {"run_id": None, "random_seed": 0, "overwrite": False},
            "paths": {"output_root": "runs_tables"},
            "dwell_tasks": [
                {
                    "task": "task5_pro",
                    "window_data_dir": "demo/window_data",
                    "video_feats_dir": "demo/video_feats",
                }
            ],
        },
    )

    pre_cfg = tmp_path / "pre.yml"
    _write_yaml(
        pre_cfg,
        {
            "paths": {"input_glob": "raw/*.json"},
            "condition_rules": [{"pattern": "x", "condition": "y", "video_length_seconds": 1.0}],
        },
    )

    ar_cfg = tmp_path / "ar.yml"
    _write_yaml(
        ar_cfg,
        {
            "run": {"run_id": "a", "sap_name": "s", "sap_version": "v"},
            "paths": {"feature_long_csv": "x.csv", "run_level_csv": "y.csv"},
            "filters": {
                "tasks_keep": ["task5_pro"],
                "groups_keep": ["Control"],
                "sex_keep": ["Female", "Male"],
                "design_videos": ["S1"],
            },
            "confirmatory": {
                "contrasts": [["Control", "Control"]],
                "endpoints": {
                    "ep": {
                        "table": "run_level",
                        "successes_col": "a",
                        "trials_col": "b",
                        "predictors": ["Group"],
                    }
                },
            },
        },
    )

    unified_cfg_path = tmp_path / "full.yml"
    _write_yaml(
        unified_cfg_path,
        {
            "run": {"output_root": str(tmp_path / "runs_pipeline"), "run_id": "u2"},
            "pipeline": {"steps": ["build_tables"]},
            "preprocess_jspsych": {"enabled": True, "config_path": str(pre_cfg)},
            "build_tables": {
                "enabled": True,
                "config_path": str(bt_cfg),
                "use_preprocess_windows": True,
            },
            "analysis_ready": {"enabled": False, "config_path": str(ar_cfg)},
        },
    )

    cfg = load_config(unified_cfg_path)
    with pytest.raises(PipelineError):
        _ = run_pipeline(config_path=unified_cfg_path, cfg=cfg)


def test_unified_pipeline_writes_failed_manifest_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bt_cfg = tmp_path / "bt.yml"
    _write_yaml(
        bt_cfg,
        {
            "run": {"run_id": None, "random_seed": 0, "overwrite": False},
            "paths": {"output_root": "runs_tables"},
            "dwell_tasks": [
                {
                    "task": "task5_pro",
                    "window_data_dir": "demo/window_data",
                    "video_feats_dir": "demo/video_feats",
                }
            ],
        },
    )

    unified_cfg_path = tmp_path / "full.yml"
    _write_yaml(
        unified_cfg_path,
        {
            "run": {
                "output_root": str(tmp_path / "runs_pipeline"),
                "run_id": "u_fail",
                "random_seed": 0,
                "overwrite": False,
            },
            "pipeline": {"steps": ["build_tables"]},
            "preprocess_jspsych": {"enabled": False, "config_path": "unused_pre.yml"},
            "build_tables": {
                "enabled": True,
                "config_path": str(bt_cfg),
                "use_preprocess_windows": False,
            },
            "analysis_ready": {"enabled": False, "config_path": "unused_ar.yml"},
        },
    )

    def fake_build_tables(*, config_path: Path) -> Path:  # noqa: ARG001
        raise RuntimeError("forced-build-failure")

    import alabwebgazer.derive_tables.pipeline as bt_pipeline

    monkeypatch.setattr(bt_pipeline, "run_pipeline", fake_build_tables)

    cfg = load_config(unified_cfg_path)
    with pytest.raises(RuntimeError, match="forced-build-failure"):
        _ = run_pipeline(config_path=unified_cfg_path, cfg=cfg)

    run_dir = tmp_path / "runs_pipeline" / "u_fail"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_step_id"] == "build_tables"
    assert "forced-build-failure" in manifest["error"]
