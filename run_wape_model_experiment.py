"""Run a separate, leakage-safe WAPE comparison on the frozen modelling panel.

This is deliberately an experimental runner.  It reads the approved modelling
CSV, reuses the project's chronological rolling-origin folds, and writes only
to a caller-selected directory below ``data/experiments``.  It never writes to
PostgreSQL or replaces production validation, model-selection, forecast, or
dashboard artefacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.model_validation import (
    PredictionRecord,
    build_rolling_origin_folds,
    district_threshold,
    evaluate_predictions,
)
from app.services.modelling_dataset import HORIZONS, DATASET_PATH, SUMMARY_PATH
from app.services.models.pooled_tree_validation import (
    POOLED_CATEGORICAL_FEATURES,
    POOLED_NUMERIC_FEATURES,
)

RANDOM_STATE = 24024
CURRENT_CLIMATE_COLUMNS = (
    "rainfall_current",
    "temperature_current",
    "humidity_current",
)
FOURIER_COLUMNS = (
    "season_sin_1",
    "season_cos_1",
    "season_sin_2",
    "season_cos_2",
)
LONG_HISTORY_COLUMNS = (
    "cases_lag_26",
    "cases_lag_52",
    "cases_rolling_mean_26",
    "cases_rolling_observed_count_26",
)
FORBIDDEN_COLUMNS = frozenset(
    {
        "population",
        "incidence_per_1000",
        "facility_reporting_completeness",
        "time_series_completeness",
        "target_cases_h4",
        "target_cases_h8",
        "target_cases_h12",
        "target_epi_year_h4",
        "target_epi_year_h8",
        "target_epi_year_h12",
        "target_epi_week_h4",
        "target_epi_week_h8",
        "target_epi_week_h12",
        "target_week_start_date_h4",
        "target_week_start_date_h8",
        "target_week_start_date_h12",
    }
)


@dataclass(frozen=True)
class FeatureVariant:
    key: str
    description: str
    add_current_climate: bool = False
    add_fourier_seasonality: bool = False
    add_long_history: bool = False

    @property
    def numeric_features(self) -> tuple[str, ...]:
        values = list(POOLED_NUMERIC_FEATURES)
        if self.add_current_climate:
            values.extend(CURRENT_CLIMATE_COLUMNS)
        if self.add_fourier_seasonality:
            values.extend(FOURIER_COLUMNS)
        if self.add_long_history:
            values.extend(LONG_HISTORY_COLUMNS)
        return tuple(values)


FEATURE_VARIANTS = (
    FeatureVariant(
        key="approved_core",
        description="Published 28-feature production contract; control comparison.",
    ),
    FeatureVariant(
        key="core_plus_issue_climate_and_seasonality",
        description=(
            "Approved core plus climate measured by the issue week and two annual "
            "Fourier harmonics. All extra inputs are available at the forecast origin."
        ),
        add_current_climate=True,
        add_fourier_seasonality=True,
    ),
    FeatureVariant(
        key="core_plus_long_history",
        description=(
            "Approved core plus 26- and 52-week malaria history calculated only "
            "from calendar weeks before the forecast origin."
        ),
        add_long_history=True,
    ),
    FeatureVariant(
        key="core_plus_long_history_climate_and_seasonality",
        description=(
            "Approved core plus calendar-aligned 26/52-week malaria history, "
            "issue-week climate, and two annual Fourier harmonics; all inputs "
            "are available at the forecast origin."
        ),
        add_current_climate=True,
        add_fourier_seasonality=True,
        add_long_history=True,
    ),
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    parameters: dict[str, Any]


MODEL_SPECS = (
    ModelSpec(
        key="random_forest_control",
        label="Random Forest control",
        family="random_forest",
        parameters={
            "n_estimators": 64,
            "max_depth": 16,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        },
    ),
    ModelSpec(
        key="xgboost_control",
        label="XGBoost control",
        family="xgboost",
        parameters={
            # The production-selected count configuration, held fixed so this
            # experiment isolates the added origin-week features.
            "n_estimators": 96,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
    ),
    ModelSpec(
        key="negative_binomial_glm",
        label="Negative Binomial GLM",
        family="negative_binomial_glm",
        parameters={
            "seasonal_terms": 2,
            "dispersion": "training-fold method-of-moments",
            "district_fixed_effects": False,
        },
    ),
    ModelSpec(
        key="poisson_glm",
        label="Poisson GLM",
        family="poisson_glm",
        parameters={"seasonal_terms": 2, "district_fixed_effects": False},
    ),
    ModelSpec(
        key="extra_trees",
        label="Extra Trees",
        family="extra_trees",
        parameters={
            "n_estimators": 256,
            "max_depth": None,
            "max_features": 0.7,
            "min_samples_leaf": 2,
        },
    ),
    ModelSpec(
        key="hist_gradient_boosting_poisson",
        label="Histogram Gradient Boosting (Poisson)",
        family="hist_gradient_boosting_poisson",
        parameters={
            "loss": "poisson",
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
    ),
)


def estimate_negative_binomial_alpha(values: np.ndarray) -> float:
    """Estimate NB2 overdispersion only from the current training-fold outcomes."""
    if len(values) < 2:
        return 1e-6
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    if mean <= 0:
        return 1e-6
    return max((variance - mean) / (mean * mean), 1e-6)


def validate_experiment_contract() -> None:
    """Protect the experiment against target leakage and accidental feature drift."""
    for variant in FEATURE_VARIANTS:
        values = set(POOLED_CATEGORICAL_FEATURES) | set(variant.numeric_features)
        overlap = sorted(values & FORBIDDEN_COLUMNS)
        if overlap:
            raise ValueError(f"Experiment variant {variant.key} includes forbidden columns: {overlap}")
        if any(name.startswith("target_") for name in values):
            raise ValueError(f"Experiment variant {variant.key} includes a target column.")
        if any("wind" in name.lower() for name in values):
            raise ValueError(f"Experiment variant {variant.key} includes an unapproved wind variable.")


def read_frozen_modelling_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Read the frozen CSV without requiring a live Flask/Database context."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Frozen modelling dataset is unavailable: {path}")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Frozen modelling dataset summary is unavailable: {SUMMARY_PATH}")
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Frozen modelling dataset summary is invalid: {SUMMARY_PATH}") from exc
    if not summary.get("csv_written"):
        raise ValueError("Frozen modelling dataset summary does not confirm a written CSV.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Frozen modelling dataset has no rows: {path}")
    return add_long_history_features(rows)


def add_long_history_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add calendar-aligned historical predictors without imputing past outcomes.

    Each value is keyed by district and week start date.  A row can therefore
    access only prior observed weeks, even though the frozen CSV is read in one
    pass.  Gaps remain missing and are handled by the fold-fitted predictor
    imputer, just like the existing lag features.
    """
    observed_cases: dict[tuple[str, datetime.date], float] = {}
    parsed_dates: list[datetime.date] = []
    for row in rows:
        try:
            origin_date = datetime.fromisoformat(str(row["week_start_date"])).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Frozen modelling dataset has an invalid district-week key.") from exc
        raw_cases = row.get("confirmed_malaria_cases")
        if raw_cases not in {None, ""}:
            try:
                value = float(raw_cases)
            except (TypeError, ValueError) as exc:
                raise ValueError("Frozen modelling dataset has an invalid observed case value.") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError("Frozen modelling dataset has an invalid observed case value.")
            observed_cases[(str(row["district_id"]), origin_date)] = value
        parsed_dates.append(origin_date)

    augmented_rows: list[dict[str, Any]] = []
    for row, origin_date in zip(rows, parsed_dates):
        district_id = str(row["district_id"])
        augmented = dict(row)
        for weeks, column in ((26, "cases_lag_26"), (52, "cases_lag_52")):
            value = observed_cases.get((district_id, origin_date - timedelta(weeks=weeks)))
            augmented[column] = "" if value is None else str(value)
        prior_values = [
            observed_cases.get((district_id, origin_date - timedelta(weeks=weeks)))
            for weeks in range(1, 27)
        ]
        available_values = [value for value in prior_values if value is not None]
        augmented["cases_rolling_observed_count_26"] = str(len(available_values))
        augmented["cases_rolling_mean_26"] = (
            "" if not available_values else str(float(np.mean(available_values)))
        )
        augmented_rows.append(augmented)
    return augmented_rows


def build_feature_frame(rows: list[dict[str, Any]], variant: FeatureVariant) -> pd.DataFrame:
    """Create an origin-week feature matrix without reading target fields."""
    frame = pd.DataFrame(index=range(len(rows)))
    for column in variant.numeric_features:
        if column in FOURIER_COLUMNS:
            continue
        frame[column] = pd.to_numeric([row.get(column) for row in rows], errors="coerce")

    if variant.add_fourier_seasonality:
        weeks = pd.to_numeric([row.get("epi_week") for row in rows], errors="coerce")
        radians = 2 * math.pi * weeks / 53.0
        frame["season_sin_1"] = np.sin(radians)
        frame["season_cos_1"] = np.cos(radians)
        frame["season_sin_2"] = np.sin(2 * radians)
        frame["season_cos_2"] = np.cos(2 * radians)

    for column in POOLED_CATEGORICAL_FEATURES:
        frame[column] = [None if row.get(column) in {None, ""} else str(row[column]) for row in rows]
    return frame[list(variant.numeric_features) + list(POOLED_CATEGORICAL_FEATURES)]


def _tree_preprocessor(variant: FeatureVariant, *, dense: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]),
                list(variant.numeric_features),
            ),
            (
                "district",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
                    ]
                ),
                list(POOLED_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def _glm_design(
    train_x: pd.DataFrame,
    validation_x: pd.DataFrame,
    variant: FeatureVariant,
    *,
    include_district_fixed_effects: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]
            ),
            list(variant.numeric_features),
        )
    ]
    if include_district_fixed_effects:
        transformers.append(
            (
                "district",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", drop="first", sparse_output=False
                            ),
                        ),
                    ]
                ),
                list(POOLED_CATEGORICAL_FEATURES),
            )
        )
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    train_design = np.asarray(preprocessor.fit_transform(train_x), dtype=float)
    validation_design = np.asarray(preprocessor.transform(validation_x), dtype=float)
    return np.column_stack([np.ones(len(train_design)), train_design]), np.column_stack(
        [np.ones(len(validation_design)), validation_design]
    )


def fit_predict(
    spec: ModelSpec,
    variant: FeatureVariant,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    target_column: str,
) -> np.ndarray:
    """Fit one pre-specified count model on a fold and return nonnegative predictions."""
    train_x = build_feature_frame(train_rows, variant)
    validation_x = build_feature_frame(validation_rows, variant)
    train_y = np.asarray([float(row[target_column]) for row in train_rows], dtype=float)

    if spec.family in {"random_forest", "extra_trees"}:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.pipeline import Pipeline

        estimator_cls = RandomForestRegressor if spec.family == "random_forest" else ExtraTreesRegressor
        estimator = estimator_cls(
            **spec.parameters,
            n_jobs=2,
            random_state=RANDOM_STATE,
        )
        model = Pipeline(
            [("preprocess", _tree_preprocessor(variant, dense=False)), ("model", estimator)]
        )
        predictions = model.fit(train_x, train_y).predict(validation_x)
    elif spec.family == "xgboost":
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            **spec.parameters,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=2,
            tree_method="hist",
            verbosity=0,
        )
        model = Pipeline(
            [("preprocess", _tree_preprocessor(variant, dense=False)), ("model", estimator)]
        )
        predictions = model.fit(train_x, train_y).predict(validation_x)
    elif spec.family == "hist_gradient_boosting_poisson":
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline

        estimator = HistGradientBoostingRegressor(**spec.parameters, random_state=RANDOM_STATE)
        model = Pipeline(
            [("preprocess", _tree_preprocessor(variant, dense=True)), ("model", estimator)]
        )
        predictions = model.fit(train_x, train_y).predict(validation_x)
    elif spec.family in {"negative_binomial_glm", "poisson_glm"}:
        import statsmodels.api as sm

        train_design, validation_design = _glm_design(
            train_x,
            validation_x,
            variant,
            include_district_fixed_effects=bool(spec.parameters.get("district_fixed_effects")),
        )
        if spec.family == "negative_binomial_glm":
            family = sm.families.NegativeBinomial(alpha=estimate_negative_binomial_alpha(train_y))
        else:
            family = sm.families.Poisson()
        fitted = sm.GLM(train_y, train_design, family=family).fit(maxiter=200, disp=0)
        predictions = fitted.predict(validation_design)
    else:  # pragma: no cover - static model specification guard
        raise ValueError(f"Unsupported experimental model: {spec.family}")

    return np.maximum(np.asarray(predictions, dtype=float), 0.0)


def _prediction_row(
    *,
    spec: ModelSpec,
    variant: FeatureVariant,
    horizon: int,
    fold_number: int,
    row: dict[str, Any],
    predicted_value: float,
    threshold: Decimal,
) -> tuple[PredictionRecord, dict[str, Any]]:
    actual = Decimal(str(row[f"target_cases_h{horizon}"]))
    predicted = Decimal(str(predicted_value)).quantize(Decimal("0.000001"))
    score = (predicted - threshold).quantize(Decimal("0.000001"))
    record = PredictionRecord(
        district_id=str(row["district_id"]),
        horizon=horizon,
        origin_date=datetime.fromisoformat(str(row["week_start_date"])).date(),
        target_date=datetime.fromisoformat(str(row[f"target_week_start_date_h{horizon}"])).date(),
        actual=actual,
        predicted=predicted,
        threshold=threshold,
        actual_surge=actual > threshold,
        predicted_surge=score > 0,
        surge_score=score,
    )
    output = {
        "model_key": spec.key,
        "model_name": spec.label,
        "feature_variant": variant.key,
        "horizon": horizon,
        "validation_fold": fold_number,
        "district_id": row["district_id"],
        "district_name": row.get("district_name", ""),
        "origin_week": row["week_start_date"],
        "target_week": row[f"target_week_start_date_h{horizon}"],
        "observed_cases": str(actual),
        "predicted_cases": str(predicted),
        "district_surge_threshold": str(threshold),
        "predicted_surge_score": str(score),
    }
    return record, output


def run_experiment(
    rows: list[dict[str, Any]],
    *,
    max_folds: int = 12,
    horizons: tuple[int, ...] = HORIZONS,
    feature_variants: tuple[FeatureVariant, ...] = FEATURE_VARIANTS,
    model_specs: tuple[ModelSpec, ...] = MODEL_SPECS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run all pre-specified models and return only experimental evidence."""
    validate_experiment_contract()
    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    folds_by_horizon = {
        horizon: build_rolling_origin_folds(rows, horizon=horizon, max_folds=max_folds)
        for horizon in horizons
    }

    for variant in feature_variants:
        for spec in model_specs:
            model_key = f"{spec.key}__{variant.key}"
            model_result: dict[str, Any] = {
                "model_name": spec.label,
                "model_family": spec.family,
                "parameters": spec.parameters,
                "feature_variant": variant.description,
                "horizons": {},
            }
            for horizon, folds in folds_by_horizon.items():
                records: list[PredictionRecord] = []
                failures: list[str] = []
                horizon_predictions: list[dict[str, Any]] = []
                target_column = f"target_cases_h{horizon}"
                for fold_number, fold in enumerate(folds, start=1):
                    try:
                        predictions = fit_predict(
                            spec,
                            variant,
                            fold.trainable_rows,
                            fold.validation_rows,
                            target_column,
                        )
                        if len(predictions) != len(fold.validation_rows):
                            raise RuntimeError("Prediction length does not match validation rows.")
                        for row, predicted_value in zip(fold.validation_rows, predictions):
                            threshold = district_threshold(
                                fold.train_rows,
                                str(row["district_id"]),
                                target_epi_week=int(row[f"target_epi_week_h{horizon}"]),
                            )
                            record, output = _prediction_row(
                                spec=spec,
                                variant=variant,
                                horizon=horizon,
                                fold_number=fold_number,
                                row=row,
                                predicted_value=float(predicted_value),
                                threshold=threshold,
                            )
                            records.append(record)
                            horizon_predictions.append(output)
                    except Exception as exc:  # noqa: BLE001 - preserve a failed candidate as evidence
                        failures.append(f"fold {fold_number}: {type(exc).__name__}: {exc}")

                if failures:
                    model_result["horizons"][str(horizon)] = {
                        "status": "failed",
                        "completed_folds": len(folds) - len(failures),
                        "expected_folds": len(folds),
                        "failures": failures,
                    }
                    continue
                count_metrics, surge_metrics = evaluate_predictions(records)
                model_result["horizons"][str(horizon)] = {
                    "status": "success",
                    "completed_folds": len(folds),
                    "expected_folds": len(folds),
                    "count_metrics": count_metrics,
                    "threshold_warning_metrics": surge_metrics,
                }
                prediction_rows.extend(horizon_predictions)
            results[model_key] = model_result

    metadata = {
        "experiment_type": "prespecified_exploratory_wape_model_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "dataset_rows": len(rows),
        "horizons": list(horizons),
        "max_folds": max_folds,
        "validation_method": "shared chronological rolling-origin validation",
        "no_random_shuffle": True,
        "outcomes_imputed": False,
        "production_artifacts_modified": False,
        "clinical_candidate_data_used": False,
        "clinical_candidate_data_exclusion_reason": (
            "Read-only audit found implausible magnitudes and cross-field arithmetic "
            "contradictions; it is not fit for model integration."
        ),
        "feature_variants": [
            {
                "key": variant.key,
                "description": variant.description,
                "categorical_features": list(POOLED_CATEGORICAL_FEATURES),
                "numeric_features": list(variant.numeric_features),
            }
            for variant in feature_variants
        ],
        "models": [
            {"key": spec.key, "label": spec.label, "parameters": spec.parameters}
            for spec in model_specs
        ],
        "interpretation_rule": (
            "Lower WAPE is better.  100 minus WAPE is not reported as prediction accuracy. "
            "Results are exploratory because the same historical validation evidence is "
            "used to compare candidates; no model may replace production without a "
            "locked confirmatory evaluation."
        ),
        "results": results,
    }
    return metadata, prediction_rows


def _select_items(raw: str, items: tuple[Any, ...], *, label: str) -> tuple[Any, ...]:
    """Select a comma-separated subset by its stable ``key`` without guessing."""
    requested = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not requested:
        raise ValueError(f"At least one {label} is required.")
    available = {item.key: item for item in items}
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return tuple(available[key] for key in requested)


def _select_horizons(raw: str) -> tuple[int, ...]:
    try:
        selected = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Horizons must be comma-separated integers.") from exc
    if not selected or any(horizon not in HORIZONS for horizon in selected):
        raise ValueError(f"Horizons must be a non-empty subset of {HORIZONS}.")
    return selected


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "model_key",
        "model_name",
        "feature_variant",
        "horizon",
        "validation_fold",
        "district_id",
        "district_name",
        "origin_week",
        "target_week",
        "observed_cases",
        "predicted_cases",
        "district_surge_threshold",
        "predicted_surge_score",
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/prespecified_wape_models_20260729"),
        help="Experimental output directory; production paths are never used.",
    )
    parser.add_argument("--max-folds", type=int, default=12)
    parser.add_argument(
        "--horizons",
        default="4,8,12",
        help="Comma-separated horizons. Use one horizon per bounded execution chunk.",
    )
    parser.add_argument(
        "--models",
        default=",".join(spec.key for spec in MODEL_SPECS),
        help="Comma-separated experimental model keys.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(variant.key for variant in FEATURE_VARIANTS),
        help="Comma-separated feature-variant keys.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_experiment_contract()
    selected_horizons = _select_horizons(args.horizons)
    selected_models = _select_items(args.models, MODEL_SPECS, label="model key")
    selected_variants = _select_items(args.variants, FEATURE_VARIANTS, label="feature variant key")
    if args.dry_run:
        print("Dry run: validated experimental contract only.")
        print(f"Models: {', '.join(spec.label for spec in selected_models)}")
        print(f"Feature variants: {', '.join(variant.key for variant in selected_variants)}")
        print(f"Horizons: {', '.join(str(horizon) for horizon in selected_horizons)}")
        return 0

    started = time.perf_counter()
    rows = read_frozen_modelling_dataset(DATASET_PATH)
    payload, predictions = run_experiment(
        rows,
        max_folds=args.max_folds,
        horizons=selected_horizons,
        feature_variants=selected_variants,
        model_specs=selected_models,
    )
    payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "validation_predictions.csv"
    _atomic_write_json(summary_path, payload)
    _atomic_write_predictions(predictions_path, predictions)
    print(f"Wrote {summary_path}")
    print(f"Wrote {predictions_path}")
    for key, result in payload["results"].items():
        values = []
        for horizon, block in result["horizons"].items():
            if block["status"] == "success":
                values.append(f"h{horizon} WAPE={block['count_metrics']['wape']}")
            else:
                values.append(f"h{horizon} FAILED")
        print(f"{key}: {', '.join(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
