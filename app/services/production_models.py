"""Final production model fitting using data available through one issue week."""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.services.models.baseline import _baseline_prediction
from app.services.models.pooled_tree_validation import (
    POOLED_FEATURES,
    PooledTreeConfig,
    _build_pipeline,
    _feature_frame,
    _fit_surge_classifier,
    _positive_class_weight,
    _positive_probabilities,
    _prediction_contributions,
    _surge_target_array,
    _target_array,
)
from app.services.models.sarimax import (
    SARIMAX_EXOGENOUS_FEATURES,
    _forecast_fold,
)


class ProductionModelError(RuntimeError):
    """Raised when a final production model cannot be fitted honestly."""


def fit_pooled_production_model(
    *,
    config: PooledTreeConfig,
    rows: list[dict[str, Any]],
    origin_rows: list[dict[str, Any]],
    issue_date: date,
    horizon: int,
    model_card_path: Path,
    staged_artifact_path: Path,
    final_artifact_path: Path,
    dataset_signature: dict[str, Any],
) -> dict[str, Any]:
    """Fit one final pooled preprocessing/model pipeline and serialize it."""
    parameters = load_validated_parameters(
        model_card_path=model_card_path,
        model_name=config.model_name,
        horizon=horizon,
    )
    classifier_parameters, probability_cutoff, operational_surge_method = load_validated_classifier_spec(
        model_card_path=model_card_path,
        model_name=config.model_name,
        horizon=horizon,
    )
    target_column = f"target_cases_h{horizon}"
    target_date_column = f"target_week_start_date_h{horizon}"
    eligible_by_date = [
        row
        for row in rows
        if not _missing(row.get(target_date_column))
        and date.fromisoformat(str(row[target_date_column])) <= issue_date
    ]
    training_rows = [
        row for row in eligible_by_date if not _missing(row.get(target_column))
    ]
    if not training_rows:
        raise ProductionModelError(
            f"No observed production labels are available for {config.model_name} h{horizon}."
        )
    if not origin_rows:
        raise ProductionModelError(
            f"No issue-week rows are available for {config.model_name} h{horizon}."
        )

    train_x = _feature_frame(training_rows)
    train_y = _target_array(training_rows, target_column)
    train_surge_y = _surge_target_array(rows, training_rows, horizon)
    origin_x = _feature_frame(origin_rows)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline = _build_pipeline(config, parameters)
        pipeline.fit(train_x, train_y)
        raw_predictions = pipeline.predict(origin_x)
        classifier, constant_probability = _fit_surge_classifier(
            config=config,
            parameters=classifier_parameters,
            train_x=train_x,
            train_y=train_surge_y,
        )
        surge_probabilities = (
            _positive_probabilities(classifier, origin_x)
            if classifier is not None
            else [constant_probability] * len(origin_rows)
        )
    transformed_names = [
        str(name)
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]
    preprocessing = _preprocessing_manifest(pipeline, transformed_names)
    pipeline.production_metadata_ = {
        "model_name": config.model_name,
        "model_scope": "national_pooled",
        "issue_week_start_date": issue_date.isoformat(),
        "horizon": horizon,
        "predictors": list(POOLED_FEATURES),
        "selected_parameters": parameters,
        "selected_classifier_parameters": classifier_parameters,
        "surge_probability_cutoff": probability_cutoff,
        "operational_surge_method": operational_surge_method,
        "surge_positive_class_weight": _positive_class_weight(train_surge_y),
        "training_rows": len(training_rows),
        "latest_observed_target_week": max(
            str(row[target_date_column]) for row in training_rows
        ),
        "latest_training_feature_week": max(
            str(row["week_start_date"]) for row in training_rows
        ),
        "dataset_signature": dataset_signature,
        "transformed_feature_names": transformed_names,
        "training_only_preprocessing": True,
        "validation_estimators_reused": False,
    }
    pipeline.surge_classifier_ = classifier
    pipeline.surge_constant_probability_ = constant_probability
    pipeline.surge_probability_cutoff_ = probability_cutoff
    import joblib

    staged_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, staged_artifact_path, compress=3)
    model_signature = file_signature(staged_artifact_path)
    contributions = _prediction_contributions(
        config, pipeline, origin_x, transformed_names
    )
    forecasts: list[dict[str, Any]] = []
    for index, (row, raw_prediction, surge_probability) in enumerate(
        zip(origin_rows, raw_predictions, surge_probabilities)
    ):
        raw_value = float(raw_prediction)
        if not math.isfinite(raw_value):
            raise ProductionModelError(
                f"{config.model_name} produced a non-finite forecast for {row['district_id']}."
            )
        clipped = raw_value < 0
        predicted = max(0.0, raw_value)
        missing_count = sum(_missing(row.get(feature)) for feature in POOLED_FEATURES)
        forecasts.append(
            {
                "district_id": str(row["district_id"]),
                "predicted_cases": predicted,
                "prediction_interval_lower": None,
                "prediction_interval_upper": None,
                "model_status": "success",
                "fallback_status": "none",
                "feature_availability_status": (
                    "complete" if missing_count == 0 else "imputed_predictors"
                ),
                "missing_predictor_count": missing_count,
                "forecast_clipped_nonnegative": clipped,
                "surge_probability": (
                    float(surge_probability)
                    if operational_surge_method == "dedicated_classifier"
                    else None
                ),
                "classifier_probability": float(surge_probability),
                "predicted_surge_flag": (
                    float(surge_probability) >= probability_cutoff
                    if operational_surge_method == "dedicated_classifier"
                    else None
                ),
                "surge_probability_cutoff": probability_cutoff,
                "operational_surge_method": operational_surge_method,
                "explanation": contributions[index] if contributions else None,
            }
        )
    return {
        "model_name": config.model_name,
        "model_scope": "national_pooled",
        "horizon": horizon,
        "status": "success",
        "training_row_count": len(training_rows),
        "missing_outcome_exclusions": len(eligible_by_date) - len(training_rows),
        "latest_training_week": max(
            str(row[target_date_column]) for row in training_rows
        ),
        "latest_training_feature_week": max(
            str(row["week_start_date"]) for row in training_rows
        ),
        "predictors": list(POOLED_FEATURES),
        "selected_parameters": parameters,
        "selected_classifier_parameters": classifier_parameters,
        "surge_probability_cutoff": probability_cutoff,
        "preprocessing": preprocessing,
        "dataset_signature": dataset_signature,
        "model_signature": model_signature,
        "model_artifact_path": str(final_artifact_path),
        "fit_status": "success",
        "fallback_information": "none",
        "limitations": [
            "No prediction interval is emitted because the fitted tree regressor does not provide an approved calibrated interval.",
            "District identity is represented by training-only one-hot encoding.",
        ],
        "warnings": sorted({str(item.message) for item in captured}),
        "transformed_feature_names": transformed_names,
        "forecasts": forecasts,
    }


def baseline_production_forecast(
    *,
    district_rows: list[dict[str, Any]],
    issue_date: date,
    target_epi_week: int,
) -> dict[str, Any]:
    """Apply the approved deterministic district baseline rule."""
    predicted = _baseline_prediction(
        district_rows,
        origin_date=issue_date,
        target_epi_week=target_epi_week,
    )
    value = float(predicted)
    if not math.isfinite(value):
        raise ProductionModelError("Baseline produced a non-finite forecast.")
    return {
        "predicted_cases": max(0.0, value),
        "prediction_interval_lower": None,
        "prediction_interval_upper": None,
        "model_status": "success",
        "fallback_status": "deterministic_rule",
        "feature_availability_status": "complete",
        "missing_predictor_count": 0,
        "forecast_clipped_nonnegative": value < 0,
        "explanation": {
            "rule": "historical same-epi-week count, else trailing four observed counts",
        },
    }


def fit_sarimax_production_model(
    *,
    district_id: str,
    district_rows: list[dict[str, Any]],
    origin_row: dict[str, Any],
    horizon: int,
    checkpoint_path: Path,
    staged_artifact_path: Path,
    final_artifact_path: Path,
    dataset_signature: dict[str, Any],
) -> dict[str, Any]:
    """Fit one district-horizon SARIMAX using the validated specification."""
    selected_order, selected_seasonal_order = load_validated_sarimax_specification(
        checkpoint_path=checkpoint_path,
        district_id=district_id,
        horizon=horizon,
    )
    fit = _forecast_fold(
        training_rows=district_rows,
        validation_row=origin_row,
        horizon=horizon,
        selected_order=selected_order,
        selected_seasonal_order=selected_seasonal_order,
    )
    if fit["status"] != "success":
        return {
            "model_name": "SARIMAX",
            "model_scope": "district_specific",
            "district_id": district_id,
            "horizon": horizon,
            "status": "failed",
            "fit_status": "failed",
            "selected_parameters": {
                "order": list(selected_order),
                "seasonal_order": list(selected_seasonal_order),
            },
            "fallback_information": fit.get("fallback_status", "failed"),
            "failure_reason": fit.get("warning_or_failure_reason"),
            "forecasts": [],
        }
    state = {
        "model_name": "SARIMAX",
        "model_scope": "district_specific",
        "district_id": district_id,
        "horizon": horizon,
        "selected_order": list(selected_order),
        "selected_seasonal_order": list(selected_seasonal_order),
        "final_order": fit["order"],
        "final_seasonal_order": fit["seasonal_order"],
        "fallback_status": fit["fallback_status"],
        "fitted_state": fit.get("fitted_state"),
        "training_rows": len(district_rows),
        "observed_training_rows": sum(
            not _missing(row.get("confirmed_malaria_cases")) for row in district_rows
        ),
        "dataset_signature": dataset_signature,
        "origin_exogenous_policy": (
            "Issue-week lagged climate values are held constant across forecast steps; no future climate rows are read."
        ),
    }
    staged_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    staged_artifact_path.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )
    model_signature = file_signature(staged_artifact_path)
    missing_count = sum(
        _missing(origin_row.get(feature)) for feature in SARIMAX_EXOGENOUS_FEATURES
    )
    return {
        "model_name": "SARIMAX",
        "model_scope": "district_specific",
        "district_id": district_id,
        "horizon": horizon,
        "status": "fallback" if fit["fallback_status"] != "none" else "success",
        "training_row_count": len(district_rows),
        "missing_outcome_exclusions": sum(
            _missing(row.get("confirmed_malaria_cases")) for row in district_rows
        ),
        "latest_training_week": str(district_rows[-1]["week_start_date"]),
        "latest_training_feature_week": str(district_rows[-1]["week_start_date"]),
        "predictors": list(SARIMAX_EXOGENOUS_FEATURES),
        "selected_parameters": {
            "order": list(selected_order),
            "seasonal_order": list(selected_seasonal_order),
            "final_order": fit["order"],
            "final_seasonal_order": fit["seasonal_order"],
        },
        "preprocessing": {
            "numeric_imputation": "district training-history median",
            "standardization": "district training-history mean and scale",
            "future_exogenous_policy": "hold issue-week lagged climate values constant",
        },
        "dataset_signature": dataset_signature,
        "model_signature": model_signature,
        "model_artifact_path": str(final_artifact_path),
        "fit_status": "success",
        "fallback_information": fit["fallback_status"],
        "limitations": [
            "Future lagged climate is approximated by holding issue-week values constant.",
            "The interval is a model-based 95% SARIMAX interval conditional on the held exogenous assumption.",
        ],
        "warnings": [fit.get("warning_or_failure_reason")]
        if fit.get("warning_or_failure_reason")
        else [],
        "forecasts": [
            {
                "district_id": district_id,
                "predicted_cases": float(fit["predicted"]),
                "prediction_interval_lower": fit.get("prediction_interval_lower"),
                "prediction_interval_upper": fit.get("prediction_interval_upper"),
                "model_status": "success",
                "fallback_status": fit["fallback_status"],
                "feature_availability_status": (
                    "complete" if missing_count == 0 else "imputed_predictors"
                ),
                "missing_predictor_count": missing_count,
                "forecast_clipped_nonnegative": fit["forecast_clipped_nonnegative"],
                "explanation": {
                    "order": fit["order"],
                    "seasonal_order": fit["seasonal_order"],
                    "origin_climate_held_constant": True,
                },
            }
        ],
    }


def load_validated_parameters(
    *, model_card_path: Path, model_name: str, horizon: int
) -> dict[str, Any]:
    if model_card_path is None:
        raise ProductionModelError(f"Validated model card path is missing for {model_name}.")
    try:
        payload = json.loads(model_card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionModelError(
            f"Validated model card is unavailable or invalid: {model_card_path}."
        ) from exc
    if payload.get("status") != "success" or payload.get("model_name") != model_name:
        raise ProductionModelError(
            f"Validated model card is not successful for {model_name}: {model_card_path}."
        )
    if payload.get("model_scope") != "national_pooled":
        raise ProductionModelError(f"{model_name} model card does not confirm pooled scope.")
    if payload.get("predictors_used") != list(POOLED_FEATURES):
        raise ProductionModelError(f"{model_name} model card predictor contract has changed.")
    selections = payload.get("selected_parameters_by_horizon") or {}
    parameters = selections.get(str(horizon), selections.get(horizon))
    if not isinstance(parameters, dict) or not parameters:
        raise ProductionModelError(
            f"{model_name} model card has no selected parameters for h{horizon}."
        )
    return dict(parameters)


def load_validated_classifier_spec(
    *, model_card_path: Path, model_name: str, horizon: int
) -> tuple[dict[str, Any], float, str]:
    try:
        payload = json.loads(model_card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionModelError(
            f"Validated model card is unavailable or invalid: {model_card_path}."
        ) from exc
    if payload.get("status") != "success" or payload.get("model_name") != model_name:
        raise ProductionModelError(
            f"Validated model card is not successful for {model_name}: {model_card_path}."
        )
    parameters_by_horizon = payload.get(
        "selected_classifier_parameters_by_horizon"
    ) or {}
    cutoffs_by_horizon = payload.get("selected_probability_cutoff_by_horizon") or {}
    methods_by_horizon = payload.get("operational_surge_method_by_horizon") or {}
    parameters = parameters_by_horizon.get(
        str(horizon), parameters_by_horizon.get(horizon)
    )
    cutoff = cutoffs_by_horizon.get(str(horizon), cutoffs_by_horizon.get(horizon))
    method = methods_by_horizon.get(str(horizon), methods_by_horizon.get(horizon))
    if not isinstance(parameters, dict) or not parameters:
        raise ProductionModelError(
            f"{model_name} model card has no surge-classifier parameters for h{horizon}."
        )
    try:
        parsed_cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ProductionModelError(
            f"{model_name} model card has no valid surge-probability cutoff for h{horizon}."
        ) from exc
    if not 0 < parsed_cutoff < 1:
        raise ProductionModelError(
            f"{model_name} surge-probability cutoff is outside (0, 1) for h{horizon}."
        )
    if method not in {"dedicated_classifier", "count_threshold"}:
        raise ProductionModelError(
            f"{model_name} model card has no valid operational surge method for h{horizon}."
        )
    return dict(parameters), parsed_cutoff, str(method)


def load_validated_sarimax_specification(
    *, checkpoint_path: Path, district_id: str, horizon: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionModelError(
            f"Validated SARIMAX checkpoint is unavailable or invalid: {checkpoint_path}."
        ) from exc
    job = payload.get("job") or {}
    if (
        job.get("district_id") != district_id
        or int(job.get("horizon", -1)) != horizon
        or job.get("status") not in {"success", "fallback", "partial_success"}
    ):
        raise ProductionModelError(
            f"SARIMAX checkpoint does not contain a usable {district_id} h{horizon} specification."
        )
    order = tuple(int(value) for value in job.get("selected_order") or [])
    seasonal = tuple(int(value) for value in job.get("selected_seasonal_order") or [])
    if len(order) != 3 or len(seasonal) != 4:
        raise ProductionModelError(
            f"SARIMAX checkpoint specification is invalid for {district_id} h{horizon}."
        )
    return order, seasonal


def file_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def _preprocessing_manifest(pipeline, transformed_names: list[str]) -> dict[str, Any]:
    preprocess = pipeline.named_steps["preprocess"]
    numeric_imputer = preprocess.named_transformers_["numeric"].named_steps["imputer"]
    district_encoder = preprocess.named_transformers_["district"].named_steps["onehot"]
    return {
        "numeric_imputation": "median fitted only on production training rows",
        "numeric_imputer_statistics": [float(value) for value in numeric_imputer.statistics_],
        "missing_indicators": True,
        "district_encoding": "one-hot fitted only on production training rows; unknown ignored",
        "district_categories": [str(value) for value in district_encoder.categories_[0]],
        "transformed_feature_names": transformed_names,
        "outcome_imputation": False,
    }


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")
