# Bounded surge-comparison completion plan

Recorded after the user's scope/closure addendum, before these classifier fits.
Earlier count results and January-June 2026 outcomes have already been inspected.
This is exploratory development completion, not a new preregistered holdout.

- Existing RF and XGBoost classifier families only; their existing three
  classifier candidates and seven probability cutoffs (0.30/0.40/0.45/0.50/
  0.55/0.60/0.70). Select by F2, recall, average precision, precision, specificity,
  then stable existing order, on the earliest original fold's eight-week inner
  validation only. No extra candidates or threshold refinement after outer scores.
- Full original predictors versus history/time/district only, at 4/8/12 weeks.
  Original twelve chronological outer targets only; no new 92-week classifier
  expansion. All training labels stop at outer O-1; the inner training labels
  stop at the earliest inner feature origin minus one week. All case predictors
  and interpolation anchors obey O-1. Each training binary event is reconstructed
  from its historical origin's available Q75 reference, excluding 2022.
- For available March-June 2026 targets, fit history-only classifiers once with
  targets through 22 December 2025, the snapshot's last 2025 ISO week. Keep weights,
  probability cutoffs and the December threshold reference frozen. Use existing
  2026 case reports as lagged forecast inputs, never as training labels. Climate
  is unavailable in that snapshot, so no full-climate 2026 result is claimed.
- Add one strict history-only RF count baseline per horizon for matched 2026
  comparison, using the count track's pre-existing earliest-fold O-1 RF parameters (tuning_rf_h*_delay1.json). This is a
  fixed-parameter reduced-feature baseline, not a newly tuned count search.
- 144 classifier outer jobs, six frozen classifier jobs and three frozen count
  jobs; 36 candidate fits for the twelve inner selections. One worker, three
  estimator threads, after the existing tree workers finish; 90-minute total
  budget, signature-checked checkpoints, failures/unrun jobs retained. No second
  search pass. No other families, data acquisition, dashboard changes or manuscript.

Compare warning F2/recall/precision, TP/FP/FN, alert rate, workload and average
precision. Pair identical outcomes/references. Original twelve-target uncertainty
uses district and origin resampling. Four-/eight-week temporal blocks are reported
for the fourteen later targets as a dependence sensitivity, with their short-span
limitations. Neither source supplies an untouched or prospective validation claim.
No acceptable operational false-alert ceiling has been established.

Conclude convincing statistical improvement, inconclusive evidence or absent
improvement per horizon, with operational acceptability stated separately. If
improvement is not convincing, retain the defensible corrected existing approach
as the research baseline and stop searching. This round ends with the verdict,
supervisor matrix, supplementary register, revision plan and verified handoff.

For a provisional recommendation, screen history-only candidates on development F2 intervals and non-increased development false alerts, choose the highest development F2 among eligible candidates, then confirm/reject that one choice on later data. Do not switch to another candidate because its 2026 result looks better. Already inspected 2026 remains exploratory; confirmation is not restored holdout protection.
