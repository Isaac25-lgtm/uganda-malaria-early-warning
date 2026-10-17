# Supervisor response matrix — evidence and student actions

S01–S20 follow the order of the twenty comments in the supplied supervisor PDF. This is an evidence/action checklist, not replacement thesis prose. Original dissertation PDFs are preserved.

| Item | Required response | Evidence or remaining action |
|---|---|---|
| S01 | Abstract: false alarms | `legacy_recomputed_metrics.csv`, `warning_workload_cost_sensitivity.csv`: use false alerts/total warnings with the correct denominator and evaluation version. Student adds the limitation. |
| S02 | Remove supervisor/second signatory signatures and dates | Student edits the authoritative Word working copy and obtains fresh signatures after presentation. Original PDF unchanged. |
| S03 | One numbered specific-objectives section | Student makes the structural edit. |
| S04 | Remove unnecessary spaces throughout | Student performs document layout pass; no reflow of the authoritative PDF attempted. |
| S05 | Move limitations to Chapter 6 | Student makes the structural edit and checks cross-references. |
| S06 | Poster appendix; check Jude/Bob requirements | A review poster and standalone figures support preparation; student confirms required template and writes/reviews final text. No colleague was contacted by the agent. |
| S07 | Objective 2 plural/models and selection | Clarify comparison of four candidates and subsequent count/warning selection. Student final wording. |
| S08 | Counts versus population-standardized incidence | Counts represent reported burden/caseload, not per-person risk. Pooled WAPE weights burden; district distributions and own-district thresholds provide context, but do not replace population adjustment. |
| S09 | District encoding | Actual tree inputs: 28 raw predictors; one-hot district identity fitted on training rows, unknown categories ignored. Numeric imputation/missing indicators fit only on training data. Canonical joins still require valid district IDs. |
| S10 | Q75/window sensitivity | `threshold_sensitivity.csv`, `06_threshold_sensitivity`: Q70/75/80 × ±1/2/3; fallback counts and include-2022 sensitivity. Definitions change labels, not count forecasts. |
| S11 | Adaptations versus replications | `publication_notes.md`, `frozen_adapted_methods_rescored.csv`, original adaptation register. Models remain adaptations. Newly accessible Zinszer full text supersedes the earlier access limitation. |
| S12 | Twelve-week temporal coverage | Per-origin results, original-origin uncertainty, December sensitivity and 92-week follow-up. Weekly observations are dependent; use temporal block uncertainty and do not call all district-weeks independent. |
| S13 | F2 and cost trade-off | `four_model_cost_sensitivity.csv`: F1/F2 and FN:FP cost ratios 1/2/4/8, common rows and workload. F2 emphasizes recall but is not an empirically established monetary loss function. |
| S14 | Missingness/exclusions by year/district | `missingness_district_year.csv`, `missingness_year.csv`, `missingness_position_audit.csv`: 247 absent rows, six adjudicated outcomes, 228 leading/19 internal absences. 2021 has zero absent rows and two adjudicated errors. Causes are not inferred. |
| S15 | Strict common sample for SARIMAX | `four_model_common_sample.csv`, corrected SARIMAX diagnostics, paired comparisons. Report insufficient-history failures, fallback and its nonseasonal specification. |
| S16 | Precision/recall/false alerts alongside F2 | `03_warning_workload`, common-model and cost tables. Distinguish binary threshold warnings from heuristic colored risk categories. |
| S17 | Small ARIMAX comparison: units and uncertainty | The submitted 0.33% is a relative reduction, correctly distinguished in the thesis from 0.129 WAPE percentage points. Original paired intervals include zero. See original benchmark tables and `corrected_rf_adapted_literature_comparisons.csv`; corrected RF changes the 8-week count point ranking. |
| S18 | Importance bias | Held-out grouped permutation diagnostics, plus ablation. Correlated inputs and within-origin permutation limit interpretation; constant time inputs cannot be assessed by this permutation scheme. |
| S19 | History-only versus history+climate | Twelve-origin fixed-parameter RF ablation and the labelled exploratory 92-week paired comparator. Incremental predictive value is distinct from causation. |
| S20 | Intended users/usability | Real HTTP/browser/software checks are completed engineering checks. Intended district-health-user testing and operational utility remain unvalidated; student states this explicitly. |

Numerical items are complete only when the final handoff/result-verification records mark their scenarios complete. Manual dissertation actions remain listed rather than being silently claimed finished.
