# Current corrected baseline

The first-round corrected baseline is preserved; round-two availability scenarios are separate scientific specifications. The recorded Git HEAD is `13fbb2e6941e61a2fe59407530a5b85150790b9b`; source corrections remain uncommitted as found at round-two start. No new application correction, commit, tag, push or model promotion occurs in this round.

| Model | Weeks | Original WAPE % | Corrected WAPE % |
| --- | --- | --- | --- |
| rf | 4 | 34.3113 | 34.3113 |
| rf | 8 | 38.5616 | 39.7440 |
| rf | 12 | 39.6427 | 39.6427 |
| xgb | 4 | 36.4397 | 36.4397 |
| xgb | 8 | 40.3821 | 40.3517 |
| xgb | 12 | 40.7035 | 39.2143 |

Source: first-round `legacy_recomputed_metrics.csv`, `master_metrics.csv` and decomposed `legacy_corrected_comparison.csv`. Original RF count metrics are unchanged at 4/12 weeks to reporting precision; eight-week corrected tuning changes its count result. Warning changes also include the explicitly versioned 52/53-week reference policy. Original 12w XGBoost has the lower corrected WAPE point estimate; its paired interval versus RF includes zero.

This is a selected-source freeze and SHA manifest, not a full disk backup. `baseline_manifest.json` hashes current source and first-round artifacts; `baseline_source/` copies selected source. Trained models, raw surveillance and large rasters are not recopied. The initial thesis freeze in the first-round folder remains separate. New source, environment and model execution controls are in this folder.
