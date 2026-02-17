# Parsed Requirements

## Source Inputs
- Manuscript methods and results narrative
- Supplementary methods and quality-control details
- Statistical analysis plan (SAP) drafts and implementation notes
- Reviewer/meta-review requests used for hardening and robustness scope
- Legacy reference implementations used for preprocessing and table-building parity

## Missing Inputs (Tracked)
- Finalized dataset schema document with locked dtypes and allowed value domains
- Final confirmatory output inventory (table/figure shells with acceptance checks)
- Final deviation log between preregistration and implemented confirmatory analyses
- Final decision note for primary inferential engine where alternatives remain

## Conflict Policy
1. Confirmatory SAP decisions take precedence when manuscript wording conflicts with executable analysis scope.
2. Reviewer-requested hardening is treated as required unless it contradicts locked confirmatory scope.
3. Unresolved conflicts are tracked as explicit TODO-CONFLICT items and must be documented in traceability.

## Key Conflicts / Ambiguities
- `TODO-CONFLICT-001`: Quadrant assignment narrative differs between documents (axis-wise vs 4-cluster framing).
- `TODO-CONFLICT-002`: Primary emphasis differs across drafts (model-derived weights vs dwell-time endpoints).
- `TODO-CONFLICT-003`: Inferential family language differs (Gaussian mixed models vs denominator-aware binomial families).
- `TODO-CONFLICT-004`: Sidedness and multiplicity language differs across preregistration drafts and review guidance.

## Extracted Requirement Set (Normalized)

### Manuscript / Supplement Core

| ID | Requirement | Priority |
|---|---|---|
| MS-001 | Preserve reproducibility: rerunnable workflows with deterministic artifacts. | Must |
| MS-002 | Apply consistent group assignment and explicit handling of ambiguous labels. | Must |
| MS-003 | Ingest timestamped WebGazer `t/x/y` streams with variable sampling-rate tolerance. | Must |
| MS-004 | Support scripted feature streams (speaker, distraction, averted-gaze states). | Must |
| MS-005 | Include pre/post quadrant-image quality checks as auditable artifacts. | Should |
| MS-006 | Enforce trial-level exclusion/quality thresholds via explicit config. | Must |
| MS-007 | Preserve preprocessing sequence parity (median filter, drift, border, quadrant, resample). | Must |
| MS-008 | Keep model-based attention-weight analyses available as optional/extended path. | Could |
| MS-009 | Persist model diagnostics and fit metadata when model steps are executed. | Should |
| MS-010 | Support mixed-effect confirmatory analyses with participant-level dependence handling. | Must |
| SM-001 | Track participant/video exclusion reasons in auditable summaries. | Must |
| SM-002 | Capture QC components (sampling, missingness, attention checks, calibration quality). | Must |
| SM-003 | Support robust sensitivity checks for denominator, missingness, and binning choices. | Must |

### SAP-Normalized Confirmatory Scope

| ID | Requirement | Priority |
|---|---|---|
| SAP-001 | Confirmatory scope anchored to task-specific endpoints and valid-conditioned estimands. | Must |
| SAP-002 | Use sufficient-statistic columns (`dwell_bins`, `n_present_any_valid`, `n_switches`, `n_contig_steps`). | Must |
| SAP-003 | Implement denominator-aware inferential models or equivalent robust alternatives. | Must |
| SAP-004 | Report marginal effect sizes and uncertainty intervals for planned contrasts. | Must |
| SAP-005 | Apply explicit multiplicity correction over confirmatory families. | Must |
| SAP-006 | Lock config flags used in confirmatory claims and expose them in run artifacts. | Must |
| SAP-007 | Include robustness/sensitivity runs for denominator, QC, and cohort stratification decisions. | Must |

### Reviewer-Driven Hardening

| ID | Requirement | Priority |
|---|---|---|
| REV-001 | Reconcile denominator accounting with attrition and exclusion reporting. | Must |
| REV-002 | Treat missingness as a first-class analysis and reporting component. | Must |
| REV-003 | Provide drift-correction and preprocessing sensitivity analyses. | Must |
| REV-004 | Clarify and validate quadrant assignment uncertainty handling. | Must |
| REV-005 | Enforce explicit inferential assumptions for bounded outcomes. | Must |
| REV-006 | Include unit-of-analysis and repeated-measure handling explicitly in model specs. | Must |
| REV-007 | Publish reproducibility appendix content (seeds, versions, script-to-output map). | Must |

## Ambiguities That Can Affect Correctness
- Final H1a operationalization (marginal vs speaker-conditioned state interaction) must remain explicit.
- Confirmatory contrast family boundaries must be locked before final reporting.
- Missing-bin treatment in switching metrics must be explicitly documented and tested.
- Cross-cohort stratification policy should be documented as confirmatory vs exploratory for each endpoint.
