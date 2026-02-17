from __future__ import annotations

from pathlib import Path

import pandas as pd

from alabwebgazer.preprocessing.batch import run_preprocess_pipeline


def test_preprocess_batch_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_glob = str((project_root / "examples" / "raw_demo" / "*.json").resolve())

    cfg_text = f'''
run:
  run_id: null
  random_seed: 0
  overwrite: false
  notes: ""

paths:
  input_glob: "{input_glob}"
  output_root: "{tmp_path}"
  windows_subdir: "windows"
  summary_csv: "preprocess_summary.csv"

filename_parsing:
  delimiter: "_"
  id_index: 0
  session_start: 1
  session_end: 2
  id_regex: null
  session_regex: null

trials:
  trial_type: "video-keyboard-response"
  require_webgazer_data: true

condition_rules:
  - pattern: "demo_video"
    condition: "demo"
    video_length_seconds: 5.0
    match_field: "stimname"
    case_sensitive: false
    use_regex: false

preprocess:
  median_kernel: 3
  median_method: "scipy_medfilt"
  drift_mode: "none"
  quadrant_method: "midpoint"
  allow_midpoint_fallback: false
  x_prob_threshold: 0.9
  y_prob_threshold: 0.7
  random_state: 0
  bin_seconds: 0.5
  resample_edge_policy: "half_open"
  resample_include_endpoint: false

output_naming:
  template: "videoview_window_{{id}}_{{session}}_{{condition}}.csv"
  add_trial_index_suffix: true

privacy:
  hash_subject_ids: false
  hash_salt_env: "ALABWEBGAZER_HASH_SALT"
'''
    cfg_path = tmp_path / "preprocess.yml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    run_dir = run_preprocess_pipeline(config_path=cfg_path)
    assert run_dir.exists()

    windows = list((run_dir / "windows").glob("videoview_window_*.csv"))
    assert windows, "expected at least one window csv"

    summary = run_dir / "tables" / "preprocess_summary.csv"
    assert summary.exists()
    df = pd.read_csv(summary)
    assert len(df) >= 1

    assert (run_dir / "artifacts" / "config.snapshot.yml").exists()
    assert (run_dir / "artifacts" / "run_metadata.json").exists()
    assert (run_dir / "manifest.json").exists()
