from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alabwebgazer.analysis_ready.pipeline import run_pipeline


def test_analysis_ready_end_to_end_synthetic(
    tmp_path: Path,
    synthetic_tables: tuple[Path, Path],
) -> None:
    feat_path, runs_path = synthetic_tables

    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        f"""run:
  run_id: "test_run"
  random_seed: 1
  sap_name: "SAP_implemented_all"
  sap_version: "TEST"
  notes: ""

paths:
  feature_long_csv: "{feat_path}"
  run_level_csv: "{runs_path}"
  output_root: "{tmp_path / 'runs'}"

filters:
  tasks_keep: ["task5_pro"]
  use_run_only: true
  groups_keep: ["Control", "ASD", "SPARK"]
  sex_keep: ["Female", "Male"]
  design_videos: ["S1", "S2"]

confirmatory:
  contrasts:
    - ["Control", "ASD"]
    - ["Control", "SPARK"]
    - ["ASD", "SPARK"]
  endpoints:
    H1_speaker:
      table: "feature_long"
      feature_filter: ["speaker"]
      successes_col: "dwell_bins"
      trials_col: "n_present_any_valid"
      predictors: ["Group", "Video", "Sex"]
    H2_distraction_offspeaker:
      table: "feature_long"
      feature_filter: ["distraction_offspeaker"]
      successes_col: "dwell_bins"
      trials_col: "n_present_any_valid"
      predictors: ["Group", "Video", "Sex"]
    H1a_averted_interaction:
      table: "feature_long"
      feature_filter: ["speaker_not_averted", "speaker_averted"]
      state_col: "State"
      successes_col: "dwell_bins"
      trials_col: "n_present_any_valid"
      predictors: ["Group", "State", "Group:State", "Video", "Sex"]
    H3_switching_overall:
      table: "run_level"
      successes_col: "n_switches"
      trials_col: "n_contig_steps"
      predictors: ["Group", "Video", "Sex"]

models:
  primary_backend: "glm_cluster"
  cluster_col: "SID"
  bootstrap:
    enabled: false
    n_resamples: 200
    seed: 1

multiplicity:
  method: "bh_fdr"
""",
        encoding="utf-8",
    )

    run_dir = run_pipeline(config_path=cfg, overwrite=True)

    out_csv = run_dir / "tables" / "01_confirmatory" / "confirmatory_results.csv"
    assert out_csv.exists()
    assert (run_dir / "artifacts" / "config.snapshot.yml").exists()
    assert (run_dir / "artifacts" / "run_metadata.json").exists()
    assert (run_dir / "artifacts" / "input_hashes.json").exists()
    assert (run_dir / "logs" / "analysis_ready.jsonl").exists()
    assert (run_dir / "manifest.json").exists()

    df = pd.read_csv(out_csv)
    assert len(df) == 12
    assert set(df["endpoint"]) == {
        "H1_speaker",
        "H2_distraction_offspeaker",
        "H1a_averted_interaction",
        "H3_switching_overall",
    }
    assert "q_value" in df.columns
    h1a = df.loc[df["endpoint"] == "H1a_averted_interaction"]
    assert not h1a.empty
    assert h1a["ci_low"].notna().all()
    assert h1a["ci_high"].notna().all()

    run_meta_path = run_dir / "artifacts" / "run_metadata.json"
    assert run_meta_path.exists()
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert run_meta["status"] == "completed"
    assert run_meta["steps_requested"] == ["ingest_analysis_ready", "confirmatory_models"]
    assert run_meta["executed_steps"] == ["ingest_analysis_ready", "confirmatory_models"]
    assert "SAP-IMPL-INGEST-001" in run_meta["requirements_used"]
    assert "SAP-IMPL-ENDP-001" in run_meta["requirements_used"]
