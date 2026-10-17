# Publication positioning and literature verification update

Prepared 15 October 2026. This is research feedback and a writing outline, not submission prose. The student's final text, supervisor approval and research permissions remain necessary before publication.

The strongest paper is one coherent question about the incremental value of environmental predictors beyond surveillance history, with training-history length as a supporting robustness question. Model comparison alone is a weaker contribution. Final emphasis should follow the completed evidence rather than a requirement that RF wins.

## Verified closest precedents

| Primary source | Relevant distinction for this project |
|---|---|
| [Zinszer et al., 2015](https://link.springer.com/article/10.1186/s12936-015-0758-4) | Six Ugandan sentinel facilities, weekly confirmed counts, environmental/clinical ARIMAX forecasting. Full text is accessible: training-only AIC, pre-whitening, residual diagnostics and a supplementary model table are described. The older benchmark record saying these details were inaccessible is superseded. District-level, limited-predictor direct-horizon adaptations remain adaptations. |
| [Fuller et al., 2025](https://link.springer.com/article/10.1186/s12936-025-05547-8) | National and 15-regional monthly DHIS2 forecasting with ARIMA/Prophet, 2020–2023; case definition includes clinical diagnoses. Different spatial/temporal resolution and outcome from this confirmed district-week study. National Ugandan forecasting already exists. |
| [Paasi et al., 2026](https://link.springer.com/article/10.1007/s44197-026-00562-7) | 2014–2023 district DHIS2 spatial analysis plus national monthly ARIMA projections. National/district surveillance coverage is not novel by itself. The paper explicitly frames forecasts as retrospective scenario continuation. |
| [Teklehaimanot et al., 2004, paper II](https://pmc.ncbi.nlm.nih.gov/articles/PMC535541/) | Ethiopian weekly Poisson distributed lags and four-week warning, evaluated with a different calendar and validation design. Current mean-temperature, national pooling and origin-observable adaptations cannot be called faithful replications. |
| [Merkord et al., 2017](https://link.springer.com/article/10.1186/s12936-017-1735-x) | EPIDEMIA already integrates surveillance, environmental data and warning communication. It is a system architecture precedent, not a single comparator model. A dashboard is an implementation contribution whose user utility needs evaluation. |
| [Aregawi et al., 2025](https://link.springer.com/article/10.1186/s12936-025-05351-4) | Uganda's 2022 epidemic assessed against a 2017–2021 Q75 reference. Supports studying epidemic-reference sensitivity, not automatic validation of this study's expanding reference or whole-year exclusion for every district. |
| [Zalwango et al., 2024](https://malariajournal.biomedcentral.com/counter/pdf/10.1186/s12936-024-04838-w.pdf) | Outbreak-detection thresholds in Uganda. Detection from observed counts differs from forecasting a future count and then thresholding it; constructed surge labels are not an independent outbreak register. |
| [MacLeod et al., 2019](https://git-stage.ecmwf.int/projects/ECVECTRI/repos/vectri/raw/doc/2019_malaria_forecast.pdf?at=670e79d8e963c02b21664e1506e5ee62f536ea42) | Uganda dynamical climate-driven forecasts and cost-loss evaluation exist; these are distinct from direct count regression. This retrieval located the ECMWF full text; no numerical cross-study performance equivalence is claimed. |

This focused check is not an exhaustive systematic review of the thesis's bibliography. Davis/Katwesige adaptations retain their original exploratory or sensitivity status; no unavailable thesis specification has been invented. Some NCBI requests returned bot challenges; these were not bypassed, and publisher/authoritative repository alternatives were used where available.

## Evidence to use

Lead with paired full-climate versus surveillance-history RF results, their uncertainty and operational workload. Separate fixed-parameter ablation from independently tuned reduced models; only the former was prespecified here. Present the four training starts on identical weekly targets with the warning reference held fixed. Explain data-quality differences in 2020 and 2021 without asserting COVID causation. Use corrected SARIMAX/common-sample comparisons as robustness; report its nonconvergence/fallback and insufficient-history coverage separately, alongside the constrained nonseasonal specification.

The 2026 protocol is a later-time retrospective evaluation, conditional on these outcomes not having been inspected elsewhere. The fourteen-week frozen-model comparison is surveillance-only; it cannot establish the full climate model's generalization over that period. The full-climate fixed-origin check has only one target week per horizon. Week-start origin markers do not establish the actual reporting/issuance date or elapsed operational lead time. Do not optimize or select additional specifications using these held-out outcomes and then call the resulting performance held out.

## Writing structure for the student

1. A bounded forecasting/early-warning question and practical reason to test additional climate/history.
2. Data vintages, district frame, count scale, ethics/authorization status, missingness and adjudication.
3. Exact input dates, reporting-delay assumptions, calendar policies, tuning embargo and eligibility.
4. Primary comparisons, operational definitions, paired uncertainty and coverage.
5. Quantitative findings with mixed/negative results preserved.
6. Implications for warning workload and limitations of routine surveillance.
7. Code/data availability subject to MoH permission; no credentials or restricted raw extracts released.

Full-climate 2026 weekly validation, vegetation acquisition, additional model families, forecast intervals and stakeholder utility are distinct future studies if needed. None should be presented as completed or necessary simply to add sophistication.
