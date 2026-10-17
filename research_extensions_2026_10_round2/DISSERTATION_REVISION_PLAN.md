# Dissertation revision plan

Confirmed methodological and numerical errors must be corrected consistently; the submitted values are retained only for the correction audit, not as equally valid alternative results. Student/supervisor decide how to present the correction account. No substantive redesign is justified by an inconclusive supplementary result.

# Thesis update matrix

Page numbers below explicitly distinguish PDF viewer pages from printed thesis pages. These are review instructions/evidence pointers, not replacement academic prose. The original PDF remains unchanged.

| Location | Required revision decision | Evidence/status |
| --- | --- | --- |
| Abstract and Chapter 6 | Align headline performance and selected-model claims with the supervisor-approved corrected version; explain warnings/false alerts | First-round corrected counts; this round's availability sensitivities must not silently replace them |
| Chapter 3 tuning and availability methods | Explain inner label embargo, case/label/reference cutoffs, preprocessing, exact estimator thread count and assumed source availability | `inner_boundary_audit.csv`, `worked_timelines.csv`, new source code and tuning JSON |
| PDF p51 / printed p36, Tables 4.5-4.6 | RF 8w WAPE changes 38.5616 to 39.7440; XGBoost 12w changes 40.7035 to 39.2143. Recalculate relative baseline reductions and other selected count metrics consistently | `corrections_for_dissertation.csv`; original count table and corrected first-round results. The original seasonal benchmark is preserved separately |
| PDF p52 / printed p37, model selection discussion | Separate count-WAPE point selection from warning-F2 selection; qualify constrained nonseasonal SARIMAX/fallback and unequal coverage | First-round strict common-sample table; new seasonal feasibility subset is supplementary |
| PDF p53 / printed p38, Table 4.8 and following paragraph | Correct affected warning confusion counts and metrics consistently. The statement RF was selected in all six positions no longer holds for corrected count-WAPE point ranking | RF retains higher F2; 12w XGBoost lower WAPE point estimate has zero-including paired intervals |
| PDF p54-55 / printed p39-40, Tables 4.9-4.10 and section 4.7.1 | Distinguish frozen adapted ARIMAX rescore from standardized refitting. Corrected 8w point ordering reverses; intervals include zero | First-round literature comparisons; source/convergence audit and small diagnostic rerun here. Original 0.33% relative vs 0.129 percentage-point distinction was already correct |
| PDF p70-71 / printed p55-56, Chapter 6 | Remove universal-winner implications; qualify climate evidence, prototype utility, baseline strength and reporting-vintage limitations | Climate support is exploratory; equal-information results and uncertainty here; 2026 retrospective warning limitations remain |
| Limitations and recommendations | Explain seasonal-reference fallback/support, district heterogeneity, dependency, information parity, uncertain issue times and new-experiment selection | Reference diagnostics, block-length sensitivity and matched comparisons; no calibrated outbreak probability or operational benefit established |
| Signatures/objectives/spacing/poster appendix | Manual authoritative-document actions with supervisor/template confirmation | First-round S01-S20 matrix retained; no automatic thesis/PDF rewrite |

Revised results need consistent table/figure/caption version labels throughout. Student writes/reviews final academic text. Confirmed incorrect results must be corrected; student and supervisor decide how to present the correction account, with submitted values retained only as provenance. Pending REC/MoH status and any use of 2026 results require the actual institutional decision; local tool access is not institutional approval.


# Exact locations in the submitted 100-page PDF

These are current PDF-viewer pages, verified from the submitted document's
bookmarks, headings and captions. They will change after editing. Source hash
and the extracted locator are in `results/dissertation_revision_locator.json`.
This is a revision checklist, not replacement academic prose.

| Current location | Required correction or check |
| --- | --- |
| Approval, PDF p4; ethical AI declaration, p3 | Apply the supervisor's signature/date instructions in the editable copy. Keep declarations accurate about actual assistance. |
| Abstract, p5 | Align surge model, F2/recall/precision, false-alert denominator and uncertainty with the verified correction account. Count WAPE remains supporting evidence. |
| Sections 1.3.2-1.3.4, p18 | Consolidate under one numbered Specific Objectives section and a three-item list. Correct Objective 2's model/models wording; preserve the surge-warning purpose. |
| 3.4.5 and 3.5.2, pp32-33 | Distinguish counts/burden from population-standardized incidence; document the district join. Confirm the provenance of subregion aggregation used in Figure 4.5 (p47): the current local reference has blank regional fields. |
| 3.6.1-3.6.4, pp33-34; Table 4.1, p43 | Use verified absent/excluded/interpolated denominators. Do not infer a missingness cause or silently impute outcomes. |
| 3.7.1-3.7.3, pp34-35; Table 3.2, p35 | State raw versus transformed inputs, training-only preprocessing and interpolation-anchor timing. Differentiate original predictors from supplementary case-delay scenarios. |
| 3.8, p35 | Explain Q75/window/excluded-2022 rules, broader-history fallback and reference support. Keep event definition separate from a classifier's alert-probability cutoff. |
| 3.9.2-3.9.6, pp36-37 | Describe actual constrained SARIMAX, RF/XGBoost architecture, corrected inner label embargo and exact fitting threads. Explain the bounded direct-classifier study without calling it a replay of the legacy classifier. |
| 3.10, pp38-39 | Correct chronological eligibility/availability claims. Distinguish week-start indices from actual issuance, assumed case delays from observed report arrivals, and retrospective data vintages from historical availability. |
| 3.11.1-3.11.4, pp39-40 | Prioritize surge F2 with recall, precision, FP/FN and workload; keep count selection separate. Add paired uncertainty and dependence limitations. Retain the thesis's already-correct distinction between F2 preference and an empirically established cost ratio. |
| 3.12, pp40-41; Table 3.3, p41 | Keep heuristic forecast/threshold risk colours distinct from binary surge warnings and calibrated outbreak probabilities. No new classifier is deployed automatically. |
| 3.14-3.15, p42 | Update correction/reproduction evidence and actual approval status. Software tests are not intended-user validation; pending approval must not be described as obtained. |
| 4.5, pp50-51; Tables 4.5-4.6, p51 | Correct affected count metrics and baseline-relative reductions consistently. Do not retain an incorrect count ranking to avoid revising text. |
| 4.6, p52; Table 4.7, p52; Table 4.8, p53 | Correct compatible-version warning metrics/confusions. Keep supplementary direct-classifier results concise; do not silently substitute a different target frame. |
| 4.7, p53 | Separate surge selection from WAPE selection, qualify uncertainty, and use the final bounded verdict. Remove a universal winner claim when unsupported. |
| 3.9.7, pp37-38; 4.7.1, pp54-55; Tables 4.9-4.10, pp54-55; Figure 4.10, p54 | Update corrected RF versus adapted-method comparisons in both tables and the WAPE plot. Explain adaptation/fidelity and common-sample limits. Original 0.33% relative versus 0.129 percentage-point arithmetic was correct; the corrected 8-week ordering nevertheless changes. |
| 4.8, p56 | Support variable-importance claims with the existing ablation/permutation evidence and correlation limitations, rather than treating impurity importance as causal evidence. |
| 4.9-4.10, pp56-60; Table 4.11, p57; Figures 4.11-4.15, pp57-60 | Label screenshots/forecasts as retrospective illustrative prototype artifacts. A nominal 2025W52 origin is not proof they were issued before their 2026 targets. Preserve provenance if legacy screenshots remain; do not imply new classifier deployment or user validation. |
| 5.2.2, p62; 5.3.1, p64 | Align model interpretation and literature comparison with corrected results; supplementary alternatives need not redesign the dissertation. |
| 5.4-5.5, p66 | Discuss incremental climate evidence, false alerts/misses, source timing and operational uncertainty without causal or utility inflation. |
| 5.7, pp68-69 | Move limitations into Chapter 6 as requested. One concrete placement is new 6.3 before Recommendations, renumbering current 6.3/6.3.1-6.3.3 to 6.4/6.4.1-6.4.3; update contents and cross-references. |
| 6.1.2, pp70-71; 6.1.4, p72; 6.2-6.3, pp72-73 | Correct the explicit all-horizon lowest-WAPE claim, qualify surge conclusions and preserve negative/inconclusive findings. State the retrospective scope and recommendations consistent with the stopping verdict. |
| Appendix C, Declaration of AI Use, pp81-82 | Student verifies the declaration against actual code/review/experiment assistance and institutional rules. |
| Appendix F.2-F.3, pp88-89; Table F.1, p88 | Correct configuration, thread, label-eligibility and predictor/threshold algorithms. Add supplementary classifier settings/cutoff selection only at the needed level. |
| Appendix F.4-F.5, pp90-91; Table F.2, p90 | Align adaptation/fidelity, evaluation-frame and reproducibility descriptions with the verified evidence. |
| Existing Appendix G, p93 | Check the existing poster, required template and corrected metrics; replace/update it as necessary rather than claiming the appendix is absent. |
| Appendix H, p94 | The submitted PDF explicitly states REC approval is pending. Keep the status factual; insert an actual approval document only when obtained. |
| Appendix I, pp95-98 | Keep exclusion/reference/extraction tables consistent with the verified snapshot. Source records and original PDFs are preserved. |

The remaining descriptive Figures 4.1-4.9 are not automatically regenerated:
the underlying frozen panel is unchanged. Figure 4.5's regional lineage needs
confirmation, and descriptive climate association is not incremental predictive
or causal evidence. Other figure/caption values should be checked for consistency
where they quote corrected metrics. No authoritative PDF/Word edit has been made.


Keep the main thesis surge-focused. Add direct classifiers/seasonal feasibility and other extra experiments to concise supplementary tables/appendices unless the verified verdict supports a substantive selection change. Update Abstract, Chapter 3 methods, Chapter 4 results/model-selection and Chapter 6 conclusions together. Use the full twenty-item response matrix for signatures, objective numbering, spacing, limitation placement and poster appendix. The verified exact heading/table/figure/appendix locator above identifies the current PDF positions; update page numbers after editing. The original PDF is preserved, and no student-authored academic prose is replaced by this technical plan.
