"""PRIMARY challenger: Teklehaimanot-inspired adapted Poisson distributed-lag.

Lineage-corrected (run 20260828r1): origin-week data are NOT used. The AR
moving-average uses observed cases at lags 1/2/3 (shifted back one week under
the origin-not-available convention), and the 4th-degree Almon polynomial
distributed lag uses climate at origin-lags 4-12 (rainfall) / 4-10 (temperature)
- which also matches the paper's lag windows. ERA5-Land mean temperature
substitutes original min/max. Poisson kept; overdispersion reported. No offset.
"""
from __future__ import annotations

import math
from typing import Any

from research_comparisons.literature_based_forecasting_benchmark.src import data, glm_common

METHOD_ID = "tekle_poisson_dl"
ADAPTATION_CLASS = "Teklehaimanot-inspired adapted Poisson autoregressive distributed-lag"
IS_PRIMARY = True
FAMILY = "poisson"
PDL_DEGREE = 4

AR_LAGS = (1, 2, 3)
RAIN_LAGS = tuple(range(4, 13))   # origin-lags 4..12
TEMP_LAGS = tuple(range(4, 11))   # origin-lags 4..10
CASE_LAGS = AR_LAGS
CLIMATE_LAGS = {"rainfall": RAIN_LAGS, "temperature": TEMP_LAGS}


def features(row: dict[str, Any], ctx) -> dict[str, float | None]:
    district = str(row["district_id"])
    origin = str(row["week_start_date"])
    feats: dict[str, float | None] = {}

    rain = [data.climate_at(ctx.climate_series, district, origin, k, "rainfall")[0] for k in RAIN_LAGS]
    temp = [data.climate_at(ctx.climate_series, district, origin, k, "temperature")[0] for k in TEMP_LAGS]
    rain_pdl = data.almon_terms(rain, PDL_DEGREE)
    temp_pdl = data.almon_terms(temp, PDL_DEGREE)
    for j in range(PDL_DEGREE + 1):
        feats[f"rain_pdl_{j}"] = None if rain_pdl is None else rain_pdl[j]
        feats[f"temp_pdl_{j}"] = None if temp_pdl is None else temp_pdl[j]

    ar_values = [data.observed_case_at(ctx.observed, district, origin, k)[0] for k in AR_LAGS]
    present = [v for v in ar_values if v is not None]
    feats["case_ma_123"] = sum(present) / len(present) if present else None

    feats["trend"] = data._to_float(row.get("continuous_week_index"))
    week = data._to_float(row.get("epi_week")) or 0.0
    feats["seas_sin1"] = math.sin(2 * math.pi * week / 53.0)
    feats["seas_cos1"] = math.cos(2 * math.pi * week / 53.0)
    feats["seas_sin2"] = math.sin(4 * math.pi * week / 53.0)
    feats["seas_cos2"] = math.cos(4 * math.pi * week / 53.0)
    return feats


def predict_fold(fold, horizon: int, ctx) -> dict[str, Any]:
    return glm_common.pooled_predict_fold(
        fold, horizon, ctx, lambda row: features(row, ctx), FAMILY, CASE_LAGS, CLIMATE_LAGS
    )
