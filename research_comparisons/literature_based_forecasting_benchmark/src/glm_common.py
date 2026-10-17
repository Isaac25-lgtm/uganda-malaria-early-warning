"""Shared pooled-GLM machinery for the count challengers (Poisson / NB).

Training-only imputation, standardisation, and district fixed effects. Nothing
here is selected after seeing results: the feature function and family are fixed
per method; only imputation/standardisation statistics are fit on training rows.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from research_comparisons.literature_based_forecasting_benchmark.src import data

FeatureFn = Callable[[dict[str, Any]], dict[str, float | None]]


def estimate_nb_alpha(values: np.ndarray) -> float:
    """NB2 overdispersion by method of moments, from training outcomes only."""
    if len(values) < 2:
        return 1.0
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    if mean <= 0:
        return 1.0
    return max((variance - mean) / (mean * mean), 1e-6)


def _matrix(feats: list[dict[str, float | None]], names: list[str], medians: dict[str, float]) -> np.ndarray:
    matrix = np.empty((len(feats), len(names)), dtype="float64")
    for i, feat in enumerate(feats):
        for j, name in enumerate(names):
            value = feat.get(name)
            matrix[i, j] = medians[name] if value is None else float(value)
    return matrix


def build_design(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    feature_fn: FeatureFn,
    *,
    district_fe: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_train, X_val, column_names) with a leading intercept column.

    Numeric features: median-imputed and standardised on TRAINING rows only.
    District fixed effects: one-hot on TRAINING districts (first as reference),
    unknown validation districts map to the reference (all-zero).
    """
    train_feats = [feature_fn(row) for row in train_rows]
    val_feats = [feature_fn(row) for row in val_rows]
    names = sorted({key for feat in train_feats for key in feat})

    medians: dict[str, float] = {}
    for name in names:
        present = [feat[name] for feat in train_feats if feat.get(name) is not None]
        medians[name] = float(np.median(present)) if present else 0.0

    x_train = _matrix(train_feats, names, medians)
    x_val = _matrix(val_feats, names, medians)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    columns = list(names)

    if district_fe:
        districts = sorted({str(row["district_id"]) for row in train_rows})
        fe_cols = districts[1:]  # first district is the reference level
        index = {district: j for j, district in enumerate(fe_cols)}

        def one_hot(rows: list[dict[str, Any]]) -> np.ndarray:
            block = np.zeros((len(rows), len(fe_cols)), dtype="float64")
            for i, row in enumerate(rows):
                pos = index.get(str(row["district_id"]))
                if pos is not None:
                    block[i, pos] = 1.0
            return block

        x_train = np.column_stack([x_train, one_hot(train_rows)])
        x_val = np.column_stack([x_val, one_hot(val_rows)])
        columns += [f"district={district}" for district in fe_cols]

    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_val = np.column_stack([np.ones(len(x_val)), x_val])
    return x_train, x_val, ["const", *columns]


def fit_pooled_glm(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    horizon: int,
    feature_fn: FeatureFn,
    family: str,
    *,
    district_fe: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a pooled Poisson or NB GLM and return (nonneg predictions, diagnostics)."""
    import statsmodels.api as sm

    y_train = np.array([float(row[f"target_cases_h{horizon}"]) for row in train_rows], dtype="float64")
    x_train, x_val, columns = build_design(train_rows, val_rows, feature_fn, district_fe=district_fe)

    diagnostics: dict[str, Any] = {
        "family": family,
        "n_train": int(len(y_train)),
        "n_features": int(x_train.shape[1]),
        "district_fixed_effects": district_fe,
    }
    if family == "poisson":
        fam = sm.families.Poisson()
    elif family == "nb":
        alpha = estimate_nb_alpha(y_train)
        diagnostics["nb_alpha"] = alpha
        fam = sm.families.NegativeBinomial(alpha=alpha)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported family {family!r}")

    result = sm.GLM(y_train, x_train, family=fam).fit(maxiter=200, disp=0)
    predictions = np.clip(np.asarray(result.predict(x_val), dtype="float64"), 0.0, None)
    diagnostics["converged"] = bool(getattr(result, "converged", True))
    df_resid = float(getattr(result, "df_resid", 0) or 0)
    diagnostics["pearson_dispersion"] = (
        float(result.pearson_chi2 / df_resid) if df_resid else None
    )
    diagnostics["deviance_dispersion"] = (
        float(result.deviance / df_resid) if df_resid else None
    )
    diagnostics["aic"] = float(result.aic) if result.aic == result.aic else None
    return predictions, diagnostics


def pooled_predict_fold(fold, horizon, ctx, feature_fn, family, case_lags, climate_lags) -> dict[str, Any]:
    """Fit one pooled-GLM fold and attach COMPUTED source-date lineage per prediction."""
    try:
        preds, diag = fit_pooled_glm(
            fold.trainable_rows, fold.validation_rows, horizon, feature_fn, family, district_fe=True
        )
    except Exception as exc:  # noqa: BLE001 - preserve failed fits as explicit records
        return {
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "diagnostics": {},
            "predictions": [
                {"row": row, "predicted": None, "status": "failed", "lineage": None}
                for row in fold.validation_rows
            ],
        }
    predictions = []
    for row, pred in zip(fold.validation_rows, preds):
        lin = data.row_lineage(
            str(row["district_id"]), str(row["week_start_date"]),
            case_lags=case_lags, climate_lags=climate_lags,
            observed=ctx.observed, climate_series=ctx.climate_series,
        )
        predictions.append({"row": row, "predicted": float(pred), "status": "success",
                            "lineage": data.lineage_summary(lin)})
    return {"status": "success", "failure_reason": None, "diagnostics": diag, "predictions": predictions}
