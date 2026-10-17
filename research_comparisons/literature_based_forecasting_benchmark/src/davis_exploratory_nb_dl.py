"""NON-PRIMARY (exploratory): Davis-inspired reduced distributed-lag NB.

NOT a Davis replication (omits GA driver selection, spatial clustering, remote
sensing, long lags). Lineage-corrected (run 20260828r1): origin-week data NOT
used - observed case history at lags 4/8/12 and climate at lags 4/8/12 only.
Kept OUT of the primary winner table.
"""
from __future__ import annotations

import math
from typing import Any

from research_comparisons.literature_based_forecasting_benchmark.src import data, glm_common

METHOD_ID = "davis_exploratory_nb_dl"
ADAPTATION_CLASS = "Davis-inspired reduced distributed-lag NB sensitivity analysis"
IS_PRIMARY = False
CLASSIFICATION = "non_primary_exploratory"
FAMILY = "nb"

CASE_LAGS = (4, 8, 12)
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
