# Correction gate: implementation record

The correction gate was executed under the student's instruction to proceed. The pre-correction source and selected evidence are frozen in `baseline/manifest.json`. The working application received repairs; legacy source remains in the baseline, while new evaluations use isolated extension paths.

The eight reviewed issues are repaired: chronological inner tuning gap, weekly SARIMAX calendar, tied-score average precision, warning-ranking margin, operational district warning metrics, F2 label, methods-report text and threshold edge cases. Additional review found the explicit score-type/Brier distinction and a latent future-anchor interpolation case; both are covered by regression tests.

`tests/test_research_corrections.py` has 11 substantive tests. `tests/test_research_statistics.py` has four tests for paired uncertainty and exact outcome/threshold matching. The full suites and live application checks are recorded in `results/verification/` and `results/live/`. A static template wording change needs no additional test that merely mirrors text; rendering and HTTP checks cover it.

The 52/53-week reference convention changes threshold membership. Report its effect separately from tuning and count forecasts, using the decomposed comparison table. The original RF count tables reproduce to reporting precision; the corrected 8-week inner tuning changes its count result. This is a correctness repair, not a search for improved thesis numbers.

No corrected models were promoted to the production dashboard. The saved model signatures and forecast fingerprint are preserved. Subsequent refitting or promotion is a separate operational change that needs a clear model/data version and supervisor-approved interpretation.

One repository writer at a time is advisable. A later independent reviewer should use the frozen baseline, exact result CSVs and independent verification script, rather than relying only on these notes. No second agent's verification is claimed.
