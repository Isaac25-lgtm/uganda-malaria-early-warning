# Runtime decision after pilot fits

15 October 2026. The first expanding RF fits increased from about 15 seconds in the earliest history to about 100-115 seconds at 2024 origins under the two-worker/three-thread allocation. The stricter availability contract selected the larger existing RF candidate at both observed horizons. Completing the 92 weekly primary comparisons plus the small original-target sensitivities may therefore exceed a worker's initial three-hour pass.

Permit at most one additional three-hour pass per horizon, using the exact same code, parameters, targets and signature-checked checkpoints. Record the first-pass status before resumption. Never run more than two horizon workers at once. No new models, targets or parameter candidates are added; this decision concerns measured runtime, not selecting favorable evaluation scores. Any remaining incomplete work after that bound is reported unrun. Original first-round and production artifacts remain unchanged.

All round-two tree fits and tuning use exactly three estimator threads. XGBoost can change fitted predictions when thread count changes in the installed version. Thus thread count is an estimator execution contract, not merely a speed setting. Fresh first-round reproduction instead uses its recorded two-thread tuning and six-thread outer fitting. Preserve the failed one-thread verification attempts and their diagnosis.

## Queue allocation refinement

2026-10-15T07:53:06.689811+00:00 UTC: the finalization/resume observer was restarted before it had launched any resume worker. After the initial controller has assigned every horizon, a completed horizon may use its one permitted resume pass in an otherwise idle slot while another initial horizon continues. The observer counts initial plus resumed workers and permits at most two; no horizon has overlapping writers. This changes idle-slot scheduling only, not source code for fitting, estimator threads, candidate choices, targets or per-horizon pass limits. Original controller and its active fits were not restarted.

## Explicit user surge-closure addendum

2026-10-15T11:34:55.169299+00:00: one bounded corrected RF/XGBoost classifier stage is necessary to resolve the original, not-yet-rerun classifier-selection evidence. It runs only after the tree queue ends, with one worker, three estimator threads and a ninety-minute budget. Existing three-candidate/seven-cutoff configurations only; 153 defined outer/frozen checkpoints and 36 inner candidate fits. No new families or automatic subsequent round. See SURGE_CLOSURE_PLAN.md.
