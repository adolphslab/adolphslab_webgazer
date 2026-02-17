# Notebook Preprocessing Mapping (Placeholder Implementation)

This document records how notebook functions were mapped into scaffold modules.

## Source notebooks examined
- `code/01_ET_data_quality_check.ipynb`
- `code/02_ET_data_preprocessing.ipynb`
- `code/03_Oculomotor_features.ipynb`

## Mapping
- `Med_filt_gaze` -> `src/alabwebgazer/preprocessing/filters.py::median_filter_xy`
- `Med_adjust` -> `src/alabwebgazer/preprocessing/filters.py::mean_adjust_to_screen_center`
- `Border_exclusion` -> `src/alabwebgazer/preprocessing/filters.py::border_exclusion`
- `GMM_gaze` (axis-wise clustering) -> `src/alabwebgazer/preprocessing/quadrant.py::assign_quadrants_axis`
- `ResampleMov` -> `src/alabwebgazer/preprocessing/resample.py::resample_to_bins`
- `preproc_pipeline` -> `src/alabwebgazer/preprocessing/pipeline.py::preprocess_webgazer_trial`

## Scope boundaries
- Implemented as placeholders with auditable contracts and tests.
- No claim of exact parity with all notebook edge-case behavior yet.
- End-to-end batch integration is implemented in `src/alabwebgazer/preprocessing/batch.py::run_preprocess_pipeline` (config-driven).
