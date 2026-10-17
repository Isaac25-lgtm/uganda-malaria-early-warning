"""Baseline/naive rolling-origin validation."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from app.services.model_features import build_feature_spec
from app.services.model_validation import (
    PredictionRecord,
    ValidationSummary,
    build_rolling_origin_folds,
    district_threshold,
    evaluate_predictions,
    write_validation_artifact,
)
from app.services.modelling_dataset import DATASET_PATH, HORIZONS, read_modelling_dataset

BASELINE_ARTIFACT = Path("data/quality/baseline_validation_summary.json")


def run_baseline_validation(
    *,
    horizons: tuple[int, ...] = HORIZONS,
    dry_run: bool = False,
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = BASELINE_ARTIFACT,
) -> ValidationSummary:
    rows = read_modelling_dataset(dataset_path)
    spec = build_feature_spec(list(rows[0].keys()))
    summary = ValidationSummary(
        model_name="Baseline/Naive",
        model_scope="district_specific",
        horizons=horizons,
        dry_run=dry_run,
        status="dry-run" if dry_run else "success",
        feature_columns=spec.feature_columns,
        forbidden_columns_excluded=spec.forbidden_columns_excluded,
        row_counts={"modelling_rows": len(rows)},
    )
    if dry_run:
        for horizon in horizons:
            folds = build_rolling_origin_folds(rows, horizon=horizon)
            summary.validation_cutoffs.extend(_fold_proofs(folds[:3]))
            summary.horizon_gap_proof.extend(_gap_proofs(folds[:3]))
        return summary

    for horizon in horizons:
        folds = build_rolling_origin_folds(rows, horizon=horizon)
        predictions: list[PredictionRecord] = []
        for fold in folds:
            by_district = _rows_by_district(fold.train_rows)
            for row in fold.validation_rows:
                actual = Decimal(str(row[f"target_cases_h{horizon}"]))
                threshold = district_threshold(
                    fold.train_rows,
                    row["district_id"],
                    target_epi_week=int(row[f"target_epi_week_h{horizon}"]),
                )
                predicted = _baseline_prediction(
                    by_district.get(row["district_id"], []),
                    origin_date=fold.origin_date,
                    target_epi_week=int(row[f"target_epi_week_h{horizon}"]),
                )
                predictions.append(
                    PredictionRecord(
                        district_id=row["district_id"],
                        horizon=horizon,
                        origin_date=fold.origin_date,
                        target_date=fold.target_date,
                        actual=actual,
                        predicted=predicted,
                        threshold=threshold,
                        actual_surge=actual > threshold,
                        predicted_surge=predicted > threshold,
                        surge_score=predicted - threshold,
                    )
                )
        count_metrics, surge_metrics = evaluate_predictions(predictions)
        summary.count_metrics_by_horizon[horizon] = count_metrics
        summary.surge_metrics_by_horizon[horizon] = surge_metrics
        summary.validation_cutoffs.extend(_fold_proofs(folds[:3]))
        summary.horizon_gap_proof.extend(_gap_proofs(folds[:3]))
        if predictions:
            first = predictions[0]
            summary.threshold_proof.append(
                {
                    "horizon": horizon,
                    "district_id": first.district_id,
                    "origin_date": first.origin_date.isoformat(),
                    "target_date": first.target_date.isoformat(),
                    "threshold_training_only": str(first.threshold),
                    "actual": str(first.actual),
                    "predicted": str(first.predicted),
                }
            )
    write_validation_artifact(artifact_path, summary)
    return summary


def _baseline_prediction(
    train_rows: list[dict],
    *,
    origin_date: date,
    target_epi_week: int,
) -> Decimal:
    candidates = [
        row
        for row in train_rows
        if int(row.get("epi_week", 0)) == target_epi_week
        and date.fromisoformat(str(row["week_start_date"])) <= origin_date
        and row.get("confirmed_malaria_cases") not in {None, ""}
    ]
    if candidates:
        return Decimal(str(candidates[-1]["confirmed_malaria_cases"]))
    observed = [
        Decimal(str(row["confirmed_malaria_cases"]))
        for row in train_rows
        if date.fromisoformat(str(row["week_start_date"])) <= origin_date
        and row.get("confirmed_malaria_cases") not in {None, ""}
    ]
    if not observed:
        return Decimal("0")
    tail = observed[-4:] if len(observed) >= 4 else observed
    return (sum(tail, Decimal("0")) / Decimal(len(tail))).quantize(Decimal("0.0001"))


def _rows_by_district(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["district_id"], []).append(row)
    for district_rows in grouped.values():
        district_rows.sort(key=lambda item: item["week_start_date"])
    return grouped


def _fold_proofs(folds) -> list[dict]:
    return [
        {
            "horizon": fold.horizon,
            "origin_week": fold.origin_date.isoformat(),
            "target_week": fold.target_date.isoformat(),
            "train_rows": len(fold.train_rows),
            "trainable_rows": len(fold.trainable_rows),
            "validation_rows": len(fold.validation_rows),
        }
        for fold in folds
    ]


def _gap_proofs(folds) -> list[dict]:
    return [
        {
            "horizon": fold.horizon,
            "origin": fold.origin_date.isoformat(),
            "target": fold.target_date.isoformat(),
            "gap_days": (fold.target_date - fold.origin_date).days,
            "gap_weeks": (fold.target_date - fold.origin_date).days // 7,
        }
        for fold in folds
    ]
