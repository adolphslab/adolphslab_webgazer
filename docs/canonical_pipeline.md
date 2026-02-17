# Canonical Pipeline

This repository provides three auditable stages. Each stage writes a run directory with:
- `logs/` JSONL logs
- `artifacts/` (config snapshot, run metadata, input hashes, stage-specific artifacts)
- `tables/` (stage outputs and convenience copies)
- `manifest.json` (high-level run summary)

## Stage 1: Raw WebGazer preprocessing (jsPsych JSON → window CSVs)

**Entry point:** `alabwebgazer preprocess-jspsych --config <YAML>`

**Inputs**
- Participant-level jsPsych JSON exports containing WebGazer streams (`webgazer_data` with `t,x,y`)
- Config-driven mapping: stimulus → `condition` and `video_length_seconds`

**Outputs**
- `windows/videoview_window_*.csv` with columns:
  - `Tstart`, `Tend` (seconds)
  - `window` (1..4 quadrant label)
  - `x_res`, `y_res`, `x_ratio`, `y_ratio` (optional; may be NaN depending on upstream availability)
- `tables/preprocess_summary.csv` (per-file diagnostics)

**Core steps**
- Median filtering → drift correction → border exclusion → quadrant assignment → time-bin resampling

## Stage 2: Table construction (window CSVs + feature PKLs → analysis-ready tables)

**Entry point:** `alabwebgazer build-tables --config <YAML>`

**Inputs**
- Window CSVs (Stage 1 outputs)
- Per-video feature PKLs (`feature_4quad_<key>.pkl`), each a list of 4 `(T,k)` arrays

**Outputs**
- `artifacts/dwelltime/<task>/...csv` dwell metrics per (run, feature)
- `artifacts/analysis_ready/df_mergedmain.csv`, `df_runs.csv`, and related tables
- `artifacts/table_validation.json` (schema + key-level validation report)
- Convenience copies to `tables/`

## Stage 3: Modeling (analysis-ready tables → results)

**Entry point:** `alabwebgazer run-analysis-ready --config <YAML>`

This stage reads analysis-ready CSVs and writes a separate run directory under the configured output root.
It writes:
- `logs/analysis_ready.jsonl`
- `artifacts/config.snapshot.yml`
- `artifacts/run_metadata.json` (steps, requirement IDs, input hashes, config hash)
- `artifacts/input_hashes.json`
- stage outputs under `tables/00_ingest/` and `tables/01_confirmatory/`
- backward-compatible root copies for `config.snapshot.yml`, `run_metadata.json`, and stage dirs

## Optional Unified Orchestration

**Entry point:** `alabwebgazer run --config <YAML>`

This optional runner executes the same stage implementations as the stage-specific commands,
but records one top-level run directory with deterministic stage folders:
- `stages/01_preprocess_windows/`
- `stages/02_build_tables/`
- `stages/03_analysis_ready/`

It also writes:
- `stage_configs/` (patched per-stage configs)
- `artifacts/config.snapshot.yml`
- `artifacts/run_metadata.json`
- `artifacts/input_hashes.json`
- `manifest.json`
