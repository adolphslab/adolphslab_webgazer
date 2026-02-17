# Repository Architecture

This is a standard `src/`-layout Python package.

- `src/alabwebgazer/preprocessing/`  
  Raw WebGazer preprocessing (single-trial + batch runner, config-driven).

- `src/alabwebgazer/derive_tables/`  
  Table construction pipeline plus adapter layer (`legacy_adapters.py`) around
  vendored dwell/clustering implementations.

- `src/alabwebgazer/legacy/`  
  Vendored scripts (`compute_dwelltime.py`, `clustering_dataprep.py`). Treat these as implementation dependencies; refactor only with tests.

- `src/alabwebgazer/analysis_ready/`  
  Analysis-ready modeling/reporting pipeline (separate runner + configs).

- `src/alabwebgazer/core/`  
  Shared run directory + provenance helpers, plus optional unified stage orchestration.

- `configs/`  
  YAML configuration templates.

- `examples/`  
  Synthetic/demo inputs (safe to commit).

- `tests/`  
  Unit tests and small integration tests.
