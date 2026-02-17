from __future__ import annotations

from pathlib import Path

import pandas as pd

from alabwebgazer.derive_tables.pipeline import run_pipeline


def test_build_tables_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    demo = project_root / "examples" / "demo_input"

    cfg_text = f'''
run:
  run_id: null
  random_seed: 0
  overwrite: false

paths:
  output_root: "{tmp_path}"

dwell_tasks:
  - task: "task5_pro"
    window_data_dir: "{(demo / "task5_pro" / "window_data").as_posix()}"
    video_feats_dir: "{(demo / "video_feats").as_posix()}"
    window_glob: "videoview*.csv"
    runlist_csv: "{(demo / "runlists" / "runlist_task5_pro.csv").as_posix()}"

  - task: "task5_spark"
    window_data_dir: "{(demo / "task5_spark" / "window_data").as_posix()}"
    video_feats_dir: "{(demo / "video_feats").as_posix()}"
    window_glob: "videoview*.csv"
    runlist_csv: "{(demo / "runlists" / "runlist_task5_spark.csv").as_posix()}"

dwell_options:
  presence_threshold: 0.5
  min_run: 1
  bin_seconds: 0.5
  include_joint_features: true
  derived_require_speaker_present: true
  derived_require_speaker_unique: false
  task5_require_speaker_unique: true
  include_speaker_not_averted_loose: false
  boundary_mode: "off"
  boundary_tau: 0.05
  boundary_power: 1.0
  save_bin_level: false

cluster:
  enabled: true
  pro_task: "task5_pro"
  spark_task: "task5_spark"
  use_run_only: true
  task_prefixes_keep: ["task5_"]
  groups_keep: ["Control"]
  features_keep: ["speaker", "distraction", "averted"]
  default_bin_seconds: 0.5
  add_stability_block: false
  keep_support_counts: true
  write_spearman_corr: false
  force_group_by_cohort:
    SPARK: "SPARK"
'''
    cfg_path = tmp_path / "build_tables.yml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    run_dir = run_pipeline(config_path=cfg_path)
    assert run_dir.exists()

    mergedmain = run_dir / "tables" / "df_mergedmain.csv"
    runs = run_dir / "tables" / "df_runs.csv"
    assert mergedmain.exists()
    assert runs.exists()

    df_m = pd.read_csv(mergedmain)
    df_r = pd.read_csv(runs)
    assert not df_m.empty
    assert not df_r.empty

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "artifacts" / "run_metadata.json").exists()
    assert (run_dir / "artifacts" / "table_validation.json").exists()
