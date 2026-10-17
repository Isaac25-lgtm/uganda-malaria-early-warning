# Round-two bounded analysis plan

Recorded before new round-two fits, 15 October 2026. All new analyses are exploratory retrospective development work. The attached review is context; the student's instructions govern access and scope.

## Primary questions and information contracts

1. Are simple forecasts competitive when predictor, training-label and threshold case cutoffs agree? Use the existing 92 weekly 2024–2025 targets for RF and simple forecasts under a one-week case-delay assumption. Also evaluate RF/XGBoost at the original twelve targets per horizon under delay 0, 1 and 2 assumptions. Delay 0 adds the observed origin-week count to the existing features; delay 1 retains prior-week case features but cuts training labels and threshold history at origin minus one week; delay 2 shifts case lags/rolling windows one further week. Interpolations require the right anchor at or before each feature's allowed case cutoff. Missing targets stay missing.
2. How fragile are warning references and district results? Describe reference observations/years, fallback, district/burden distributions and false/missed alerts. A prespecified reference-sufficiency sensitivity requires at least eight seasonal observations spanning three distinct years; report coverage and abstentions, not a retrospectively optimized F2. Compare temporal block lengths 4/8/12/16 weeks. These are uncertainty sensitivities, not independent confirmations.
3. Does an explicitly seasonal classical model deserve further study? Evaluate fixed direct district ridge ARX with and without annual Fourier terms/target-minus-52-week cases on the same strict weekly targets. Run a small annual SARIMA feasibility study: twelve districts selected deterministically across 2020–2023 burden ranks, four origins in 2024–2025, plain AR(1) versus AR(1) × seasonal AR(1,52), maximum 50 optimizer iterations and 60 seconds per fit. Fit once per district/origin and forecast all horizons. Nonconvergence is a failed fit; no silent replacement by another model. This subset cannot establish national superiority. Audit the existing adapted ARIMAX source and reproduce representative fits rather than rerunning all historical five-order searches.

## Fixed design

Keep 4/8/12-week targets, observed outcomes, district frame and Q75 ±2-week primary warning policy (excluding 2022). Within each case-delay scenario all compared methods share the same threshold. Across delay scenarios thresholds can differ, so separate count differences from changed event definitions. Climate retains original 4/8/12-week origin-relative lags. Actual reporting arrivals and environmental release vintages are unavailable: delay scenarios do not establish operational availability. Calendar week starts are not issuance timestamps.

For each tree model, horizon and delay, select among its existing three candidates using the earliest original fold only. Inner training labels end at earliest inner validation origin minus the assumed delay; outer labels end at outer origin minus delay. Use MAE, stable first-candidate tie handling and the existing estimator seed. The two families receive three candidates each, not identical compute. Reuse those parameters thereafter. No later targets or 2026 labels choose a configuration. New input contracts may alter hyperparameter selection; do not attribute all differences to one added predictor.

## Resource and scope limits

At most two horizon workers with three estimator threads each, three-hour per-worker wall limit and resumable signature-checked outputs. Measure runtime; retain failed/unrun jobs explicitly. Classical seasonal jobs have individual timeouts; district ridge follows a separate bounded queue step. No duplicate writers. No app code/model promotion, Git commits/pushes, new paid services or surveillance extraction. Keep all original and first-round outputs unchanged.

January–June 2026 outcomes have already been inspected. Further analyses there would be exploratory. No new 2026 full-climate holdout is claimed; no data after 30 June are accessed. Environmental products receive a feasibility/source register only in this bounded round; vegetation acquisition, new model families and true rolling-window searches are deferred rather than described as tested. Formal research approvals remain pending.

## Deliverables and stopping

Produce checkpoint registry, equal-information metrics and paired uncertainty, threshold/district diagnostics, seasonal feasibility and ARIMAX audit, source/timeline/dissertation update notes, verification results and one consolidated handoff. Preserve every attempted fit/status. Run meaningful scientific tests and independent arithmetic/temporal checks; reuse unchanged full-suite evidence explicitly. Copy the final handoff and evidence package to Downloads. Stop after these questions are answered or the recorded runtime limits are reached; do not search additional models for a favorable result.

## User priority clarification during execution

2026-10-15T08:31:44.608899+00:00: the user clarified that surge warning is primary and count WAPE can use a different model or approach. Keep the running pre-fit MAE/count experiments unchanged. Add a distinct surge-first evidence table, warning-versus-count point-choice ledger, paired warning uncertainty and always/never-alert references. This is a transparently timed interpretive amendment after the h8 count result was inspected; no new pre-registration claim, outer-data cutoff optimization, model promotion or dedicated-classifier rerun. The best thresholded count policy is not claimed to be the best surge-optimized model.
