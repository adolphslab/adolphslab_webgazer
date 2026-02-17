# Contributing

## Development Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Architecture Overview
The repository is stage-first:
1. RAW -> WINDOW in `src/alabwebgazer/preprocessing/`
2. WINDOW -> TABLES in `src/alabwebgazer/derive_tables/`
3. TABLES -> INFERENCE in `src/alabwebgazer/analysis_ready/`

The optional unified orchestration layer is in `src/alabwebgazer/core/` and calls the same stage modules.

## How To Add A New Analysis-Ready Step
1. Add a step module under `src/alabwebgazer/analysis_ready/steps/`.
2. Subclass `Step` from `src/alabwebgazer/analysis_ready/steps/base.py`.
3. Implement `run(self, ctx: RunContext) -> None`.
4. Register the step in `src/alabwebgazer/analysis_ready/steps/registry.py`.
5. Add/update tests under `tests/` (unit + one stage-level integration case).
6. Add or update requirement links in `docs/traceability_matrix.md`.

## How To Extend Other Stages
- Preprocessing logic: add modules under `src/alabwebgazer/preprocessing/` and wire them through `batch.py` / `pipeline.py`.
- Table construction logic: add adapters in `src/alabwebgazer/derive_tables/legacy_adapters.py` or new helper modules in `src/alabwebgazer/derive_tables/`.
- Unified runner behavior: update `src/alabwebgazer/core/config.py` and `src/alabwebgazer/core/pipeline.py` only after stage-level behavior is stable.

## Traceability Rule
Every new module/function that contributes to scientific output must reference at least one requirement ID (`MS-*`, `SM-*`, `SAP-*`, `REV-*`).

## Quality Gates
Run before opening a PR:
```bash
ruff check .
ruff format --check .
mypy src
pytest
python -m build
```

## Data Safety
- Never commit raw participant-level sensitive data.
- Keep secrets out of configs and logs.
- Use synthetic fixtures for tests.
