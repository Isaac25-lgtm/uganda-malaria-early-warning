"""NON-PRIMARY (sensitivity): Katwesige-inspired adapted Negative Binomial.

Lineage-corrected (run 20260828r1): origin-week data NOT used. Case history from
OBSERVED cases at lags 1/2/4/8/12; climate at lags 4/8/12 (no origin-week / no
'current'). NB/log link; national pooled + district fixed effects (training-only).
No offset, incidence, reporting-completeness predictor, outcome imputation, or
full-data screening. MLP excluded; no thesis claims reused. Kept OUT of the
primary winner table.
"""
from __future__ import annotations

import math
from typing import Any

from research_comparisons.literature_based_forecasting_benchmark.src import data, glm_common

METHOD_ID = "katwesige_nb"
ADAPTATION_CLASS = "Katwesige-inspired adapted Negative Binomial sensitivity analysis"
IS_PRIMARY = False
CLASSIFICATION = "non_primary_sensitivity"
FAMILY = "nb"

CASE_LAGS = (1, 2, 4, 8, 12)
CLIMATE_LAGS = {"rainfall": (4, 8, 12), "temperature": (4, 8, 12), "humidity": (4, 8, 12)}


def features(row: dict[str, Any], ctx) -> dict[str, float | None]:
    district = str(row["district_id"])
    origin = str(row["week_start_date"])
    feats: dict[str, float | None] = {}
    for lag in CASE_LAGS:
        feats[f"case_lag_{lag}"] = data.observed_case_at(ctx.observed, district, origin, lag)[0]
    for var, lags in CLIMATE_LAGS.items():
        for lag in lags:
            feats[f"{var}_lag_{lag}"] = data.climate_at(ctx.climate_series, district, origin, lag, var)[0]
    feats["trend"] = data._to_float(row.get("continuous_week_index"))
    week = data._to_float(row.get("epi_week")) or 0.0
    feats["seas_sin1"] = math.sin(2 * math.pi * week / 53.0)
    feats["seas_cos1"] = math.cos(2 * math.pi * week / 53.0)
    return feats


def predict_fold(fold, horizon: int, ctx) -> dict[str, Any]:
    return glm_common.pooled_predict_fold(
        fold, horizon, ctx, lambda row: features(row, ctx), FAMILY, CASE_LAGS, CLIMATE_LAGS
    )
