# Reproduce The Paper (Workflow Skeleton)

This repository is designed to support an auditable "reproduce the paper" workflow, but scientific reproduction requires:
- final locked data dictionaries and schemas
- finalized analysis-ready output inventory (table/figure shells)
- finalized modeling specifications

## Typical sequence

1) **Preprocess raw jsPsych/WebGazer exports** (if starting from raw streams):

```bash
alabwebgazer preprocess-jspsych --config configs/preprocess.yaml
```

2) **Build analysis-ready tables**:

```bash
alabwebgazer build-tables --config configs/build_tables.yaml
```

3) **Run modeling/reporting on the analysis-ready tables**:

```bash
alabwebgazer run-analysis-ready --config configs/analysis_ready.yaml --overwrite
```

## What to archive for audit

For each run directory:
- `artifacts/config.snapshot.yml`
- `artifacts/run_metadata.json`
- `artifacts/input_hashes.json`
- `manifest.json`
- all stage outputs under `windows/`, `tables/`, and `artifacts/`
