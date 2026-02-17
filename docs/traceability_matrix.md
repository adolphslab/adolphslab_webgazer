# Traceability Matrix (Engineering)

This file maps stable requirement IDs to implementation artifacts.

| Requirement ID | Description | Key config(s) | Code artifacts | Outputs | Tests |
|---|---|---|---|---|---|
| RAW-WINDOW-001 | Raw jsPsych/WebGazer → binned quadrant window CSVs | `configs/preprocess.yaml` | `alabwebgazer.preprocessing.batch.run_preprocess_pipeline` | `windows/videoview_window_*.csv` | `tests/test_preprocess_batch_smoke.py` |
| RAW-WINDOW-002 | Per-file preprocessing diagnostics | `configs/preprocess.yaml` | `alabwebgazer.preprocessing.pipeline.preprocess_webgazer_trial` | `tables/preprocess_summary.csv` | `tests/test_preprocessing_placeholder.py` |
| WINDOW-TABLES-001 | Dwell metrics from windows + features | `configs/build_tables.yaml` | `alabwebgazer.derive_tables.pipeline.run_pipeline` + `alabwebgazer.derive_tables.legacy_adapters.run_dwell_task` | `artifacts/dwelltime/<task>/*.csv` | `tests/test_build_tables_smoke.py` |
| WINDOW-TABLES-002 | Analysis-ready tables | `configs/build_tables.yaml` | `alabwebgazer.derive_tables.legacy_adapters.run_cluster_build` + `alabwebgazer.derive_tables.pipeline._validate_output_tables` | `df_mergedmain.csv`, `df_runs.csv`, `artifacts/table_validation.json` | `tests/test_build_tables_smoke.py`, `tests/test_build_tables_regression.py` |
| TABLES-INFER-001 | Modeling pipeline from analysis-ready tables | `configs/analysis_ready.yaml` | `alabwebgazer.analysis_ready.pipeline.run_pipeline` + `alabwebgazer.analysis_ready.config.load_config` | `tables/*`, `artifacts/run_metadata.json`, `artifacts/input_hashes.json` | `tests/test_analysis_ready_end_to_end.py`, `tests/test_analysis_ready_config_contract.py` |
| ORCH-E2E-001 | Optional unified run orchestration | `configs/full_pipeline.yaml` | `alabwebgazer.core.config.load_config` + `alabwebgazer.core.pipeline.run_pipeline` | `stages/*`, `stage_configs/*`, top-level `manifest.json` | `tests/test_core_unified_config.py`, `tests/test_core_unified_pipeline.py` |
| ENG-AUDIT-001 | Run manifests + config snapshot + input hashes | all | `alabwebgazer.core.run.*` and `analysis_ready/provenance.py` | `manifest.json`, `artifacts/*` | smoke tests |

See `src/alabwebgazer/requirements.py` for the canonical list of requirement IDs.
