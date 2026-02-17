# AdolphsLab WebGazer

This repository contains code and reproducible workflows for quadrant-based webcam
eye-tracking analysis (WebGazer + jsPsych) used in the Adolphs Lab.

The pipeline is organized into explicit stages:
1. Raw WebGazer/jsPsych exports -> binned quadrant windows
2. Window-level gaze outputs -> standardized analysis tables
3. Analysis-ready tables -> confirmatory inference outputs

Each stage writes auditable artifacts, including config snapshots, input hashes,
structured logs, and run manifests.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Repository Layout](#repository-layout)
  - [Configuration Files](#configuration-files)
- [Usage](#usage)
  - [Raw WebGazer Preprocessing (RAW -> WINDOW)](#raw-webgazer-preprocessing-raw---window)
  - [Table Construction (WINDOW -> TABLES)](#table-construction-window---tables)
  - [Analysis-Ready Modeling (TABLES -> INFERENCE)](#analysis-ready-modeling-tables---inference)
  - [Optional Unified End-to-End Run](#optional-unified-end-to-end-run)
- [Additional Analyses](#additional-analyses)
- [Data Policy](#data-policy)
- [Development](#development)
- [Acknowledgement](#acknowledgement)
- [Companion Library](#companion-library)

## Overview

Remote eye-tracking can be highly informative, but it also carries practical challenges:
irregular sampling, heterogeneous export formats, trial-level metadata drift, and
incomplete participant-level recordings. If these details are handled inconsistently,
downstream metrics can become difficult to reproduce or audit.

This repository addresses that problem with a stage-first, configuration-driven workflow.
In routine use, teams validate and run each stage independently, inspect outputs, then
advance to the next stage. For automation and full-trace reruns, the same stage modules
can also be executed through the optional unified orchestrator.

Bundled templates under `configs/` and fixtures under `examples/` are designed to be
runnable out of the box and to serve as reference contracts for study-specific configs.

The repository is actively maintained and designed so that additional analysis modules
can be integrated without changing the core stage contracts. This allows teams to keep
day-to-day preprocessing and table generation stable while expanding analysis-ready
endpoints in a controlled, testable way.

## Getting Started

The repository includes demo and synthetic fixtures to support quick onboarding:
- demo raw jsPsych exports in `examples/raw_demo/`
- demo Stage 2 inputs in `examples/demo_input/`
- synthetic analysis-ready fixtures in `examples/synthetic/`
- stage templates in `configs/`

### Installation

We recommend using an isolated Python environment so dependencies remain separate from
other projects and so workflow runs are easier to reproduce across collaborators.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Using `conda` (alternative):

```bash
conda create -n alabwebgazer python=3.11 -y
conda activate alabwebgazer
pip install -U pip
pip install -e ".[dev]"
```

Optional extras:

```bash
# Optional dependency set for GMM-based quadrant assignment
pip install -e ".[ml]"
```

After installation, verify the CLI entry points:

```bash
alabwebgazer --help
alabwebgazer validate-preprocess-config --help
```

### Repository Layout

- `src/alabwebgazer/preprocessing/`: RAW -> WINDOW stage
- `src/alabwebgazer/derive_tables/`: WINDOW -> TABLES stage
- `src/alabwebgazer/analysis_ready/`: TABLES -> INFERENCE stage
- `src/alabwebgazer/core/`: optional unified orchestration
- `examples/`: demo and synthetic fixtures
- `configs/`: stage and full-pipeline YAML templates
- `docs/`: pipeline contracts, provenance notes, and traceability documents

### Configuration Files

The main templates are:
- `configs/preprocess.yaml`
- `configs/build_tables.yaml`
- `configs/analysis_ready.yaml`
- `configs/full_pipeline.yaml`

`configs/default.yaml` is a reference index.

Path behavior: stage config paths resolve relative to the config file location. Bundled
templates therefore use `../examples/...` and `../runs_*`.

## Usage

The examples below follow a practical cadence used in lab workflows:
1. run bundled demo configs to verify environment and contracts
2. copy each template and adapt only study-specific fields
3. validate configs before each run
4. inspect run artifacts before moving to the next stage

This pattern is intentionally conservative. Most errors in longitudinal eye-tracking
projects come from subtle contract drift (filenames, condition mapping, or filtering
rules), and those errors are easiest to catch at stage boundaries.

### Raw WebGazer preprocessing (RAW -> WINDOW)

Use this stage when starting from participant-level jsPsych exports containing
WebGazer streams (`t/x/y` records in `webgazer_data`).

This stage performs:
- trial selection and row filtering
- confidence and border handling
- drift correction and quadrant assignment
- resampling into fixed-width bins
- one `videoview_window_*.csv` per trial

Run the bundled demo:

```bash
alabwebgazer validate-preprocess-config --config configs/preprocess.yaml
alabwebgazer preprocess-jspsych --config configs/preprocess.yaml
```

Typical outputs in `runs_preprocess/<run_id>/`:
- `windows/videoview_window_*.csv`
- `tables/preprocess_summary.csv`
- `logs/preprocess.jsonl`
- `artifacts/config.snapshot.yml`
- `artifacts/run_metadata.json`
- `artifacts/input_hashes.json`
- `manifest.json`

Narrative adaptation example:

Assume your study exports are stored under `/data/study_a/jspsych_exports/` and each
JSON filename embeds participant and session IDs. Start by copying the template, then
adjust only the fields that define your study contract.

```bash
cp configs/preprocess.yaml configs/my_preprocess.yaml
```

Update:
- `paths.input_glob` for your raw JSON location
- `filename_parsing.*` for ID/session parsing
- `condition_rules` for condition labels and video durations
- `preprocess.*` for bin width, thresholds, and edge behavior

Validate and run:

```bash
alabwebgazer validate-preprocess-config --config configs/my_preprocess.yaml
alabwebgazer preprocess-jspsych --config configs/my_preprocess.yaml
```

At this point, inspect `preprocess_summary.csv` and several window outputs before moving
to Stage 2.

Recommended checks before Stage 2:
- verify trial counts by condition match expectations
- verify window counts per trial are consistent with video duration and bin settings
- verify participant/session parsing in output filenames

### Table construction (WINDOW -> TABLES)

Use this stage when you have window CSVs and feature pickles and need standardized
analysis tables (`df_mergedmain.csv`, `df_runs.csv`).

This stage performs:
- dwell and switching metric derivation
- deterministic table assembly
- contract validation for downstream modeling

Run the bundled demo:

```bash
alabwebgazer validate-build-tables-config --config configs/build_tables.yaml
alabwebgazer build-tables --config configs/build_tables.yaml
```

Typical outputs in `runs_tables/<run_id>/`:
- `tables/df_mergedmain.csv`
- `tables/df_runs.csv`
- `artifacts/dwelltime/`
- `artifacts/analysis_ready/`
- `artifacts/table_validation.json`
- `logs/build_tables.jsonl`
- `manifest.json`

Narrative adaptation example:

Assume Stage 1 was run for two cohorts and each produced a separate `windows/` folder.
Copy the template and point each dwell task to the matching windows, feature directory,
and runlist contract.

```bash
cp configs/build_tables.yaml configs/my_build_tables.yaml
```

Edit:
- `dwell_tasks[*].window_data_dir`
- `dwell_tasks[*].video_feats_dir`
- `dwell_tasks[*].runlist_csv`
- `dwell_tasks[*].subject_metadata_csv` (if used)
- `cluster.groups_keep` for explicit cohort scope

Validate and run:

```bash
alabwebgazer validate-build-tables-config --config configs/my_build_tables.yaml
alabwebgazer build-tables --config configs/my_build_tables.yaml
```

Before Stage 3, inspect `table_validation.json` and confirm expected counts in
`df_mergedmain.csv` and `df_runs.csv`.

Recommended checks before Stage 3:
- verify `groups_keep` and `tasks_keep` filters match your analysis scope
- verify key fields (`ID`, `Group`, `Video`, `Condition`) are complete
- verify no unexpected cohort is dropped by template filters

### Analysis-ready modeling (TABLES -> INFERENCE)

Use this stage when `df_mergedmain.csv` and `df_runs.csv` are ready for confirmatory
endpoints and model summaries.

Run the bundled demo:

```bash
alabwebgazer validate-analysis-ready-config --config configs/analysis_ready.yaml
alabwebgazer run-analysis-ready --config configs/analysis_ready.yaml --overwrite
```

Typical outputs in `runs_analysis_ready/<run_id>/`:
- `logs/analysis_ready.jsonl`
- `artifacts/config.snapshot.yml`
- `artifacts/run_metadata.json`
- `artifacts/input_hashes.json`
- `tables/00_ingest/`
- `tables/01_confirmatory/confirmatory_results.csv`
- `manifest.json`

Narrative adaptation example:

Assume Stage 2 outputs are finalized for a preregistered analysis. Copy the template,
point paths to your Stage 2 outputs, then lock filters and endpoint definitions to match
the analysis plan.

```bash
cp configs/analysis_ready.yaml configs/my_analysis_ready.yaml
```

Edit:
- `paths.feature_long_csv`
- `paths.run_level_csv`
- `filters.tasks_keep`, `filters.groups_keep`, `filters.design_videos`
- `models.primary_backend` and `confirmatory.endpoints` if required by your SAP

Run:

```bash
alabwebgazer validate-analysis-ready-config --config configs/my_analysis_ready.yaml
alabwebgazer run-analysis-ready --config configs/my_analysis_ready.yaml --overwrite
```

Schema-only ingest check (without model fitting):

```bash
alabwebgazer run-analysis-ready \
  --config configs/my_analysis_ready.yaml \
  --steps ingest_analysis_ready \
  --overwrite
```

List available step IDs:

```bash
alabwebgazer list-analysis-ready-steps
```

Recommended modeling checks:
- run ingest-only first when onboarding a new dataset
- lock endpoint lists and filter settings to match your analysis plan
- archive run manifests alongside manuscript/SAP drafts

### Optional unified end-to-end run

Use unified orchestration when one top-level run folder is preferred for the entire
stage sequence.

Run bundled full pipeline:

```bash
alabwebgazer validate-run-config --config configs/full_pipeline.yaml
alabwebgazer run --config configs/full_pipeline.yaml
```

Typical outputs in `runs_pipeline/<run_id>/`:
- `stages/01_preprocess_windows/`
- `stages/02_build_tables/`
- `stages/03_analysis_ready/`
- `stage_configs/`
- `logs/pipeline.jsonl`
- `artifacts/config.snapshot.yml`
- `artifacts/run_metadata.json`
- `artifacts/input_hashes.json`
- `manifest.json`

Narrative adaptation example:

If your team wants strict stage handoff in a single command, enable:
- `build_tables.use_preprocess_windows: true`
- `analysis_ready.use_build_tables_outputs: true`

With those switches, Stage 2 consumes Stage 1 windows and Stage 3 consumes Stage 2
tables in the same orchestrated run.

Extended narrative usage example:

For a new study launch, a common pattern is to run each stage independently for one
pilot participant, review artifacts, then run the full cohort. A practical sequence is:

```bash
# 1) Stage 1 with study-specific config
alabwebgazer validate-preprocess-config --config configs/my_preprocess.yaml
alabwebgazer preprocess-jspsych --config configs/my_preprocess.yaml

# 2) Stage 2, pointing directly to Stage 1 windows
alabwebgazer validate-build-tables-config --config configs/my_build_tables.yaml
alabwebgazer build-tables --config configs/my_build_tables.yaml

# 3) Stage 3 confirmatory run
alabwebgazer validate-analysis-ready-config --config configs/my_analysis_ready.yaml
alabwebgazer run-analysis-ready --config configs/my_analysis_ready.yaml --overwrite
```

Once that pilot pass is validated, use the unified runner for full-cohort reruns and
single-folder provenance:

```bash
alabwebgazer validate-run-config --config configs/full_pipeline.yaml
alabwebgazer run --config configs/full_pipeline.yaml
```

## Additional Analyses

In addition to the core confirmatory pipeline, the repository includes expanded analysis
scope in package modules and stage outputs.

Current scope integrated into the repository:
- Advanced dwell and switching metrics in Stage 2 table derivation
- Quadrant-image calibration quality checks in preprocessing
- Confirmatory model endpoints in analysis-ready steps

Extended analysis modules are organized under package subfolders and tracked as part of
the repository roadmap. These modules have stable entry points and explicit TODO markers
for implementation details where applicable:
- `src/alabwebgazer/analysis_ready/models/multinomial.py`: model-based attention
  characterization, pseudo-R2, and variance-partition analyses
- `src/alabwebgazer/analysis_ready/models/icc.py`: ICC reliability analyses
- `src/alabwebgazer/analysis_ready/models/isc.py`: inter-subject synchrony analyses
- `src/alabwebgazer/analysis_ready/models/oculomotor.py`: oculomotor feature analyses
- `src/alabwebgazer/preprocessing/qc_extensions.py`: participant-level QC extensions
- `src/alabwebgazer/analysis_ready/models/cross_device.py`: cross-device validation

As these modules are completed, each will be exposed through config-driven
`analysis_ready` steps with reproducible run artifacts and manifest entries.

## Data Policy

- Do not commit participant-level private data.
- Keep secrets out of YAML configs and logs.
- Use environment variables for sensitive values.
- Files under `examples/` are demo or synthetic fixtures for reproducible walkthroughs.

## Development

```bash
ruff format .
ruff check .
mypy src
pytest
python -m build
```

## Acknowledgement

This repository is maintained for Adolphs Lab eye-tracking analyses, with emphasis on
reproducible preprocessing, traceable table generation, and auditable confirmatory runs.

## Companion Library

This repository is designed to be used alongside other Adolphs Lab eye-tracking
tooling, especially:
- `adolphslab_eyetracking` [cite: https://github.com/adolphslab/adolphslab_eyetracking]
