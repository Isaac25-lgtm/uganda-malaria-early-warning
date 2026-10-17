"""Frozen experiment-local fold manifest — reuses the canonical fold logic.

Does NOT redefine rolling-origin folds. Calls the canonical
``build_rolling_origin_folds`` (12 folds, expanding history, direct h4/h8/h12)
and freezes a manifest so every challenger is evaluated on identical folds.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

from app.services.model_validation import ValidationFold, build_rolling_origin_folds

from research_comparisons.literature_based_forecasting_benchmark import config as C

FOLD_MANIFEST_COLUMNS = (
    "district_id",
    "district_name",
    "horizon",
    "fold_number",
    "origin_date",
    "target_date",
    "epi_year",
    "epi_week",
    "observed_cases",
    "source_row_key",
)


def source_row_key(district_id: str, week_start_date: str, horizon: int) -> str:
    return f"{district_id}|{week_start_date}|h{horizon}"


def build_folds(rows: list[dict[str, Any]], horizon: int) -> list[ValidationFold]:
    return build_rolling_origin_folds(rows, horizon=horizon, max_folds=C.MAX_FOLDS)


def folds_by_horizon(rows: list[dict[str, Any]]) -> dict[int, list[ValidationFold]]:
    return {horizon: build_folds(rows, horizon) for horizon in C.HORIZONS}


def write_fold_manifest(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Freeze the fold manifest and return summary counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    per_horizon: dict[int, int] = {}
    for horizon in C.HORIZONS:
        folds = build_folds(rows, horizon)
        per_horizon[horizon] = len(folds)
        for fold_number, fold in enumerate(folds, start=1):
            for row in fold.validation_rows:
                district_id = str(row["district_id"])
                week = str(row["week_start_date"])
                records.append(
                    {
                        "district_id": district_id,
                        "district_name": row.get("district_name", ""),
                        "horizon": horizon,
                        "fold_number": fold_number,
                        "origin_date": week,
                        "target_date": str(row[f"target_week_start_date_h{horizon}"]),
                        "epi_year": row.get("epi_year", ""),
                        "epi_week": row.get("epi_week", ""),
                        "observed_cases": row.get(f"target_cases_h{horizon}", ""),
                        "source_row_key": source_row_key(district_id, week, horizon),
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return {
        "fold_manifest_rows": len(records),
        "folds_per_horizon": per_horizon,
        "distinct_row_keys": len({record["source_row_key"] for record in records}),
    }


def assert_target_after_origin(rows: list[dict[str, Any]]) -> None:
    """Guard: every fold's target strictly follows its origin by exactly the horizon."""
    for horizon in C.HORIZONS:
        for fold in build_folds(rows, horizon):
            if (fold.target_date - fold.origin_date).days != horizon * 7:
                raise ValueError(
                    f"Fold target/origin gap != {horizon} weeks: {fold.origin_date} -> {fold.target_date}"
                )
            for row in fold.validation_rows:
                origin = date.fromisoformat(str(row["week_start_date"]))
                target = date.fromisoformat(str(row[f"target_week_start_date_h{horizon}"]))
                if target <= origin:
                    raise ValueError(f"Validation target not after origin: {origin} -> {target}")
