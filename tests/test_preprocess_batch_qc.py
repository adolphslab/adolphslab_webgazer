from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alabwebgazer.preprocessing.batch import run_preprocess_pipeline


def _make_jspsych_export(path: Path) -> None:
    inner_w = 1000
    inner_h = 800

    def pts(x0: float, y0: float) -> list[dict[str, float]]:
        return [
            {"t": 600.0, "x": x0, "y": y0},
            {"t": 900.0, "x": x0 + 5.0, "y": y0 + 3.0},
            {"t": 1200.0, "x": x0 - 4.0, "y": y0 - 2.0},
        ]

    trials = [
        {
            "trial_type": "html-keyboard-response",
            "stimulus": "<div style='top:25%; left:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": pts(100, 100),
        },
        {
            "trial_type": "html-keyboard-response",
            "stimulus": "<div style='top:25%; right:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": pts(900, 100),
        },
        {
            "trial_type": "html-keyboard-response",
            "stimulus": "<div style='bottom:25%; left:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": pts(100, 700),
        },
        {
            "trial_type": "html-keyboard-response",
            "stimulus": "<div style='bottom:25%; right:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": pts(900, 700),
        },
        {
            "trial_type": "video-keyboard-response",
            "stimname": "demo_video_S1",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": [
                {"t": 100.0, "x": 100.0, "y": 100.0},
                {"t": 350.0, "x": 120.0, "y": 120.0},
                {"t": 600.0, "x": 800.0, "y": 120.0},
                {"t": 900.0, "x": 800.0, "y": 650.0},
                {"t": 1200.0, "x": 100.0, "y": 650.0},
            ],
        },
    ]

    path.write_text(json.dumps(trials), encoding="utf-8")


def test_preprocess_batch_writes_quadrant_qc_artifact(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    export_path = raw_dir / "demo_subj1_sess1.json"
    _make_jspsych_export(export_path)

    cfg_text = f'''
run:
  run_id: "qc_test"
  random_seed: 0
  overwrite: true
  notes: ""

paths:
  input_glob: "{str((raw_dir / '*.json').resolve())}"
  output_root: "{tmp_path}"
  windows_subdir: "windows"
  summary_csv: "preprocess_summary.csv"

filename_parsing:
  delimiter: "_"
  id_index: 1
  session_start: 2
  session_end: 3
  id_regex: null
  session_regex: null

trials:
  trial_type: "video-keyboard-response"
  require_webgazer_data: true

condition_rules:
  - pattern: "demo_video"
    condition: "demo"
    video_length_seconds: 2.0
    match_field: "stimname"
    case_sensitive: false
    use_regex: false

preprocess:
  median_kernel: 1
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

quadrant_image_qc:
  enabled: true
  q1_pattern: "top:25%; left:25%"
  q2_pattern: "top:25%; right:25%"
  q3_pattern: "bottom:25%; left:25%"
  q4_pattern: "bottom:25%; right:25%"

output_naming:
  template: "videoview_window_{{id}}_{{session}}_{{condition}}.csv"
  add_trial_index_suffix: true

privacy:
  hash_subject_ids: false
  hash_salt_env: "ALABWEBGAZER_HASH_SALT"
'''

    cfg_path = tmp_path / "preprocess_qc.yml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    run_dir = run_preprocess_pipeline(config_path=cfg_path)
    summary_csv = run_dir / "tables" / "preprocess_summary.csv"
    df = pd.read_csv(summary_csv)

    assert "qc_quadimg_raw_acc" in df.columns
    assert "qc_quadimg_preproc_acc" in df.columns
    assert "qc_quadimg_n_trials" in df.columns
    assert (df["qc_quadimg_n_trials"] >= 4).any()

    qc_json = run_dir / "artifacts" / "qc_quadrant_images_subj1_sess1.json"
    assert qc_json.exists()
