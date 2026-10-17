"""PRIMARY challenger: Zinszer-inspired adapted district-level ARIMAX.

Adapted ARIMAX / dynamic regression (no claim the paper used exactly this).
One model per district x horizon x fold.

Lineage-corrected (run 20260828r1): origin-week data are NOT used. The endog
observed-case series ends at O-1, and the exogenous climate is lagged by
(horizon + 1) so that the (h+1)-step-ahead forecast of cases at O+h requires
only climate observed at/before O-1. ARIMA(p,d,q) order is selected by AIC over
a narrow predeclared candidate set inside each fold's training history. Failed
fits remain failed records.
"""
from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Any

import numpy as np

from research_comparisons.literature_based_forecasting_benchmark.src import data

METHOD_ID = "zinszer_arimax"
ADAPTATION_CLASS = "Zinszer-inspired adapted district-level ARIMAX (dynamic regression)"
IS_PRIMARY = True
CANDIDATE_ORDERS = ((1, 0, 0), (2, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1))
EXOG_VARS = ("rainfall", "temperature")


def _weekly_frame(district, origin_iso, horizon, observed, climate_series):
    origin = date.fromisoformat(origin_iso)
    last_iso = (origin - timedelta(weeks=1)).isoformat()  # endog ends at O-1
    dweeks = [w for (d, w) in climate_series if d == district and w <= last_iso]
    if not dweeks:
        return None
    start = date.fromisoformat(min(dweeks))
    index: list[str] = []
    day = start
    while day.isoformat() <= last_iso:
        index.append(day.isoformat())
        day += timedelta(weeks=1)

    endog = np.full(len(index), np.nan, dtype="float64")
    exog = np.full((len(index), len(EXOG_VARS)), np.nan, dtype="float64")
    max_case: str | None = None
    for i, week in enumerate(index):
        value = observed.get((district, week))
        if value is not None:
            endog[i] = value
            max_case = week if max_case is None else max(max_case, week)
        for j, var in enumerate(EXOG_VARS):
            cval, _ = data.climate_at(climate_series, district, week, horizon + 1, var)
            if cval is not None:
                exog[i, j] = cval
    for j in range(len(EXOG_VARS)):
        col = exog[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
        exog[:, j] = col

    steps = horizon + 1  # forecast weeks O, O+1, ..., O+h (h+1 steps beyond O-1)
    exog_future = np.full((steps, len(EXOG_VARS)), np.nan, dtype="float64")
    max_climate: str | None = None
    for s in range(steps):
        target_week = (origin + timedelta(weeks=s)).isoformat()
        for j, var in enumerate(EXOG_VARS):
            cval, source = data.climate_at(climate_series, district, target_week, horizon + 1, var)
            exog_future[s, j] = cval if cval is not None else np.nanmedian(exog[:, j])
            if cval is not None:
                max_climate = source if max_climate is None else max(max_climate, source)
    return endog, exog, exog_future, max_case, max_climate


def _fit_forecast(endog, exog, exog_future, steps):
    import statsmodels.api as sm

    if np.isfinite(endog).sum() < 20:
        raise ValueError("insufficient observed history (<20 weeks)")
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in CANDIDATE_ORDERS:
            try:
                res = sm.tsa.SARIMAX(
                    endog, exog=exog, order=order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=0, maxiter=50, method="lbfgs")
                aic = float(res.aic)
                if aic == aic and (best is None or aic < best[0]):
                    forecast = res.forecast(steps=steps, exog=exog_future)
                    best = (aic, order, float(forecast[-1]))
            except Exception:  # noqa: BLE001
                continue
    if best is None:
        raise ValueError("all candidate orders failed to fit")
    return best[2], best[1], best[0]


def predict_fold(fold, horizon: int, ctx) -> dict[str, Any]:
    origin_iso = fold.origin_date.isoformat()
    predictions = []
    for row in fold.validation_rows:
        district = str(row["district_id"])
        try:
            frame = _weekly_frame(district, origin_iso, horizon, ctx.observed, ctx.climate_series)
            if frame is None:
                raise ValueError("no district history before origin")
            endog, exog, exog_future, max_case, max_climate = frame
            pred, order, aic = _fit_forecast(endog, exog, exog_future, horizon + 1)
            sources = [s for s in (max_case, max_climate) if s is not None]
            lineage = {
                "max_case_source_date": max_case,
                "max_climate_source_date": max_climate,
                "max_feature_source_date": max(sources) if sources else None,
                "violates_origin": bool(sources) and max(sources) > origin_iso,
            }
            predictions.append({"row": row, "predicted": max(0.0, float(pred)), "status": "success",
                                "selected_order": order, "aic": aic, "lineage": lineage})
        except Exception as exc:  # noqa: BLE001
            predictions.append({"row": row, "predicted": None, "status": "failed",
                                "failure_reason": f"{type(exc).__name__}: {exc}", "lineage": None})
    successes = sum(1 for p in predictions if p["status"] == "success")
    return {
        "status": "success" if successes else "failed",
        "failure_reason": None if successes else "all district fits failed",
        "diagnostics": {"districts": len(predictions), "successful_fits": successes,
                        "failed_fits": len(predictions) - successes},
        "predictions": predictions,
    }
