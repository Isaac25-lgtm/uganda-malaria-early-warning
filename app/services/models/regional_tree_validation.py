"""Controlled DHIS2-subregion pooling experiment for tree regressors.

This module orchestrates a matched, leakage-safe comparison between the
national-pooled Random Forest / XGBoost strategy and a DHIS2-subregion-pooled
strategy with an explicit national fallback. It never replaces production
models, forecasts or selection artefacts, never contacts DHIS2, and writes only
beneath ``data/experiments/regional_pooling/``.

Design principles
-----------------
* The shared national rolling-origin folds are built once and *scoped* to each
  subregion, so national and regional candidates use identical origin and target
  dates. Regional folds are never independently selected.
* Every regional fit reuses the single leakage-safe engine in
  :mod:`app.services.models.pooled_tree_validation`; no preprocessing, threshold,
  metric, classifier, fold or publication logic is duplicated here.
* District surge thresholds remain district-specific and training-only within a
  scoped fold; a district's own history is fully present in its subregion, so its
  threshold is identical to the national one.
* The regional-with-fallback output uses a regional prediction only where a valid
  regional candidate exists; otherwise it uses the matched national prediction and
  records the reason. National and regional forecasts are never averaged.
"""
from __future__ import annotations

import csv
import json
import platform
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from app.services.model_validation import (
    PredictionRecord,
    ValidationFold,
    build_rolling_origin_folds,
    classification_metrics,
    district_threshold,
    evaluate_predictions,
)
from app.services.modelling_dataset import DATASET_PATH, HORIZONS, read_modelling_dataset
from app.services.models.pooled_tree_validation import (
    PREDICTION_COLUMNS,
    RANDOM_STATE,
    PooledTreeConfig,
    _dataset_signature,
    publish_staged_outputs,
    run_pooled_tree_validation,
)
from app.services.models.regional_mapping import (
    REGIONAL_MAPPING_PATH,
    RegionalMapping,
    load_regional_mapping,
    write_region_mapping_csv,
)

EXPERIMENT_ROOT = Path("data/experiments/regional_pooling")
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 24024

# Regional candidates inherit the model family's production controlled thread
# count, so the experiment matches the national configuration exactly. Set this
# to an integer only to override thread count in a constrained environment;
# thread count changes parallelism, not results (RandomForest and XGBoost are
# deterministic with respect to n_jobs given a fixed random_state).
REGIONAL_CONTROLLED_THREADS: int | None = None
# Permutation importance is computed for regional runs exactly as it is for the
# national runs, so feature-importance output is directly comparable.
REGIONAL_COMPUTE_PERMUTATION_IMPORTANCE = True

REGIONAL_PREDICTION_COLUMNS = PREDICTION_COLUMNS + ("subregion", "scope_source")

SUBREGION_METRIC_COLUMNS = (
    "model_name",
    "strategy",
    "subregion",
    "horizon",
    "district_count",
    "scope_source",
    "n",
    "mae",
    "rmse",
    "mape",
    "wape",
    "r_squared",
    "surge_n",
    "tp",
    "tn",
    "fp",
    "fn",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "f2",
    "pr_auc",
)

DISTRICT_METRIC_COLUMNS = (
    "model_name",
    "strategy",
    "district_id",
    "district_name",
    "subregion",
    "horizon",
    "scope_source",
    "n",
    "mae",
    "rmse",
    "mape",
    "wape",
    "r_squared",
)

COMPARISON_ROW_COLUMNS = (
    "model_name",
    "horizon",
    "district_id",
    "district_name",
    "subregion",
    "scope_source",
    "n",
    "national_wape",
    "regional_wape",
    "relative_wape_reduction_pct",
    "national_mae",
    "regional_mae",
    "district_outcome",
)


class RegionalTreeValidationError(RuntimeError):
    """Raised when the regional-pooling experiment cannot proceed safely."""


# --------------------------------------------------------------------------- #
# Output-path helpers
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def experiment_family_dir(model_key: str, *, root: Path = EXPERIMENT_ROOT) -> Path:
    return root / model_key


def experiment_output_paths(model_key: str, *, root: Path = EXPERIMENT_ROOT) -> dict[str, Path]:
    base = experiment_family_dir(model_key, root=root)
    return {
        "validation_summary": base / "validation_summary.json",
        "validation_predictions": base / "validation_predictions.csv",
        "district_horizon_metrics": base / "district_horizon_metrics.csv",
        "subregion_horizon_metrics": base / "subregion_horizon_metrics.csv",
        "fit_diagnostics": base / "fit_diagnostics.json",
        "model_card": base / "model_card.json",
        "feature_importance": base / "feature_importance.csv",
    }


def _subregion_artifact_paths(model_key: str, subregion: str, *, root: Path = EXPERIMENT_ROOT) -> dict[str, Path]:
    base = experiment_family_dir(model_key, root=root) / "subregions" / _slug(subregion)
    return {
        "artifact_path": base / "validation_summary.json",
        "predictions_path": base / "validation_predictions.csv",
        "district_metrics_path": base / "district_horizon_metrics.csv",
        "diagnostics_path": base / "fit_diagnostics.json",
        "model_card_path": base / "model_card.json",
        "feature_importance_path": base / "feature_importance.csv",
        "checkpoint_dir": base / "checkpoints",
    }


def _national_baseline_paths(model_key: str, *, root: Path = EXPERIMENT_ROOT) -> dict[str, Path]:
    base = root / "national_baseline" / model_key
    return {
        "artifact_path": base / "validation_summary.json",
        "predictions_path": base / "validation_predictions.csv",
        "district_metrics_path": base / "district_horizon_metrics.csv",
        "diagnostics_path": base / "fit_diagnostics.json",
        "model_card_path": base / "model_card.json",
        "feature_importance_path": base / "feature_importance.csv",
        "checkpoint_dir": base / "checkpoints",
    }


# --------------------------------------------------------------------------- #
# Fold scoping
# --------------------------------------------------------------------------- #
def _scope_fold(fold: ValidationFold, district_ids: frozenset[str]) -> ValidationFold:
    """Restrict a national fold's rows to one subregion, preserving fold dates."""
    return ValidationFold(
        horizon=fold.horizon,
        origin_date=fold.origin_date,
        target_date=fold.target_date,
        train_rows=[row for row in fold.train_rows if str(row["district_id"]) in district_ids],
        trainable_rows=[row for row in fold.trainable_rows if str(row["district_id"]) in district_ids],
        validation_rows=[row for row in fold.validation_rows if str(row["district_id"]) in district_ids],
    )


def _scoped_folds_by_horizon(
    national_folds: dict[int, list[ValidationFold]],
    district_ids: frozenset[str],
    horizons: tuple[int, ...],
) -> dict[int, list[ValidationFold]]:
    return {
        horizon: [_scope_fold(fold, district_ids) for fold in national_folds.get(horizon, [])]
        for horizon in horizons
    }


def _load_dataset_and_national_folds(
    dataset_path: Path, horizons: tuple[int, ...], max_folds: int, *, rows: list[dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], dict[int, list[ValidationFold]]]:
    if rows is None:
        rows = read_modelling_dataset(dataset_path)
    if not rows:
        raise RegionalTreeValidationError(f"Modelling dataset is empty: {dataset_path}.")
    national_folds = {
        horizon: build_rolling_origin_folds(rows, horizon=horizon, max_folds=max_folds)
        for horizon in horizons
    }
    return rows, national_folds


# --------------------------------------------------------------------------- #
# Feasibility audit (computed before any fitting)
# --------------------------------------------------------------------------- #
def _validation_surge_labels(
    validation_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], horizon: int
) -> list[bool]:
    """Observed surge labels for validation rows, caching thresholds per district/week.

    ``district_threshold`` is O(train_rows) per call, so thresholds are computed
    once per (district_id, target_epi_week) and reused. Only validation rows are
    labelled here (there are at most one per district per fold), which keeps the
    pre-training audit inexpensive at national scale.
    """
    cache: dict[tuple[str, int], Decimal] = {}
    labels: list[bool] = []
    for row in validation_rows:
        target = row.get(f"target_cases_h{horizon}")
        if target in {None, ""}:
            continue
        district_id = str(row["district_id"])
        week = int(row[f"target_epi_week_h{horizon}"])
        key = (district_id, week)
        if key not in cache:
            cache[key] = district_threshold(train_rows, district_id, target_epi_week=week)
        labels.append(Decimal(str(target)) > cache[key])
    return labels


def build_feasibility_audit(
    mapping: RegionalMapping,
    national_folds: dict[int, list[ValidationFold]],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Report per subregion/horizon/fold sample sizes and fallback decisions."""
    subregion_rows: list[dict[str, Any]] = []
    for subregion in mapping.subregions:
        district_ids = frozenset(mapping.district_ids_in(subregion))
        single = mapping.is_single_district_subregion(subregion)
        for horizon in horizons:
            for fold_number, fold in enumerate(national_folds.get(horizon, []), start=1):
                scoped = _scope_fold(fold, district_ids)
                labels = _validation_surge_labels(scoped.validation_rows, scoped.train_rows, horizon)
                positives = sum(1 for label in labels if label)
                negatives = len(labels) - positives
                count_feasible = bool(scoped.trainable_rows) and bool(scoped.validation_rows)
                if single:
                    decision = "national_fallback_single_district"
                elif not count_feasible:
                    decision = "national_fallback_no_regional_targets"
                else:
                    decision = "regional_candidate"
                subregion_rows.append(
                    {
                        "subregion": subregion,
                        "horizon": horizon,
                        "validation_fold": fold_number,
                        "origin_week": fold.origin_date.isoformat(),
                        "target_week": fold.target_date.isoformat(),
                        "district_count": len(district_ids),
                        "training_rows": len(scoped.trainable_rows),
                        "validation_rows": len(scoped.validation_rows),
                        "surge_positive_rows": positives,
                        "surge_negative_rows": negatives,
                        "surge_prevalence": (round(positives / len(labels), 6) if labels else None),
                        "validation_surge_both_classes": (0 < positives < len(labels)) if labels else False,
                        "count_candidate_feasible": count_feasible,
                        "decision": decision,
                    }
                )
    return {
        "subregion_totals": {
            subregion: mapping.subregion_counts[subregion] for subregion in mapping.subregions
        },
        "poolable_subregions": list(mapping.poolable_subregions()),
        "national_fallback_subregions": list(mapping.fallback_subregions()),
        "per_subregion_horizon_fold": subregion_rows,
        "note": (
            "A regional count candidate runs only for a subregion/horizon/fold with "
            "non-empty scoped training and validation targets. Single-district "
            "subregions (Kampala) always use the national candidate. Surge decisions "
            "use the count-versus-district-threshold method; folds without both "
            "training surge classes fall back to the approved controlled behaviour."
        ),
    }


# --------------------------------------------------------------------------- #
# National baseline (reuse-or-rebuild)
# --------------------------------------------------------------------------- #
def _existing_national_paths(model_key: str) -> dict[str, Path]:
    return {
        "summary": Path(f"data/quality/{model_key}_validation_summary.json"),
        "predictions": Path(f"data/modelling/{model_key}_validation_predictions.csv"),
        "diagnostics": Path(f"data/quality/{model_key}_fit_diagnostics.json"),
    }


def _fold_key_set(national_folds: dict[int, list[ValidationFold]], horizons: tuple[int, ...]) -> set[tuple[int, str, str]]:
    keys: set[tuple[int, str, str]] = set()
    for horizon in horizons:
        for fold in national_folds.get(horizon, []):
            keys.add((horizon, fold.origin_date.isoformat(), fold.target_date.isoformat()))
    return keys


def _existing_national_is_reusable(
    model_key: str,
    national_folds: dict[int, list[ValidationFold]],
    horizons: tuple[int, ...],
    dataset_path: Path,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    paths = _existing_national_paths(model_key)
    for name, path in paths.items():
        if not path.exists():
            reasons.append(f"missing {name} artefact at {path}")
    if reasons:
        return False, reasons
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "success" or not summary.get("published"):
        reasons.append("national validation summary does not report a published success")
    diagnostics = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
    signature = _dataset_signature(dataset_path)
    if (diagnostics.get("dataset_signature") or {}).get("sha256") != signature.get("sha256"):
        reasons.append("national dataset signature does not match the current modelling dataset")
    # Fold origin/target dates must match exactly.
    expected = _fold_key_set(national_folds, horizons)
    actual: set[tuple[int, str, str]] = set()
    for cutoff in summary.get("validation_cutoffs", []):
        try:
            actual.add((int(cutoff["horizon"]), str(cutoff["origin_week"]), str(cutoff["target_week"])))
        except (KeyError, ValueError):
            continue
    actual = {key for key in actual if key[0] in horizons}
    if expected != actual:
        reasons.append("national fold origins/targets do not match the requested horizons/max-folds")
    # Prediction keys complete and unique.
    if not reasons:
        keys: set[tuple[str, str, str, str]] = set()
        n = 0
        with paths["predictions"].open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["horizon"]) not in horizons:
                    continue
                n += 1
                keys.add((row["district_id"], row["horizon"], row["validation_fold"], row["target_week"]))
        if n == 0:
            reasons.append("national predictions file has no rows for the requested horizons")
        elif len(keys) != n:
            reasons.append("national prediction keys are not unique")
    return (not reasons), reasons


def _read_predictions(path: Path, horizons: tuple[int, ...]) -> dict[tuple[int, str, str, str], dict[str, str]]:
    indexed: dict[tuple[int, str, str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            horizon = int(row["horizon"])
            if horizon not in horizons:
                continue
            key = (horizon, row["validation_fold"], str(row["district_id"]), str(row["target_week"]))
            indexed[key] = row
    return indexed


def _resolve_national_baseline(
    config: PooledTreeConfig,
    *,
    rows: list[dict[str, Any]],
    national_folds: dict[int, list[ValidationFold]],
    horizons: tuple[int, ...],
    max_folds: int,
    dataset_path: Path,
    replace: bool,
    root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    reusable, reasons = _existing_national_is_reusable(config.model_key, national_folds, horizons, dataset_path)
    if reusable:
        predictions = _read_predictions(_existing_national_paths(config.model_key)["predictions"], horizons)
        return {
            "source": "existing_production_artifact",
            "predictions_path": str(_existing_national_paths(config.model_key)["predictions"]),
            "predictions": predictions,
            "reasons": [],
        }
    # Build a fresh experimental national baseline; never touch production artefacts.
    paths = _national_baseline_paths(config.model_key, root=root)
    summary = run_pooled_tree_validation(
        config=config,
        horizons=horizons,
        dry_run=False,
        dataset_path=dataset_path,
        rows=rows,
        fixed_folds_by_horizon=national_folds,
        max_folds=max_folds,
        replace=replace,
        record_pipeline_run=False,
        progress_callback=progress_callback,
        model_scope_label="national_pooled",
        compute_permutation_importance=REGIONAL_COMPUTE_PERMUTATION_IMPORTANCE,
        **paths,
    )
    if summary.status != "success":
        raise RegionalTreeValidationError(
            f"Fresh national baseline for {config.model_name} did not succeed: {summary.status}."
        )
    predictions = _read_predictions(paths["predictions_path"], horizons)
    return {
        "source": "fresh_experimental_baseline",
        "predictions_path": str(paths["predictions_path"]),
        "predictions": predictions,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Regional-with-fallback assembly
# --------------------------------------------------------------------------- #
@dataclass
class SubregionRun:
    subregion: str
    district_ids: tuple[str, ...]
    status: str
    predictions: dict[tuple[int, str, str, str], dict[str, str]] = field(default_factory=dict)
    summary_status: str = ""
    job_counts: dict[str, int] = field(default_factory=dict)
    artifact_dir: str = ""
    warnings: list[str] = field(default_factory=list)


def _run_subregion(
    config: PooledTreeConfig,
    *,
    subregion: str,
    district_ids: frozenset[str],
    rows: list[dict[str, Any]],
    national_folds: dict[int, list[ValidationFold]],
    horizons: tuple[int, ...],
    max_folds: int,
    dataset_path: Path,
    replace: bool,
    root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> SubregionRun:
    scoped_rows = [row for row in rows if str(row["district_id"]) in district_ids]
    scoped_folds = _scoped_folds_by_horizon(national_folds, district_ids, horizons)
    paths = _subregion_artifact_paths(config.model_key, subregion, root=root)

    def _relabel(item: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback({**item, "subregion": subregion})

    summary = run_pooled_tree_validation(
        config=config,
        horizons=horizons,
        dry_run=False,
        dataset_path=dataset_path,
        rows=scoped_rows,
        fixed_folds_by_horizon=scoped_folds,
        max_folds=max_folds,
        replace=replace,
        record_pipeline_run=False,
        progress_callback=_relabel,
        model_scope_label="dhis2_subregion_pooled",
        scope_signature=f"subregion:{subregion}",
        compute_permutation_importance=REGIONAL_COMPUTE_PERMUTATION_IMPORTANCE,
        **paths,
    )
    predictions: dict[tuple[int, str, str, str], dict[str, str]] = {}
    if paths["predictions_path"].exists():
        predictions = _read_predictions(paths["predictions_path"], horizons)
    return SubregionRun(
        subregion=subregion,
        district_ids=tuple(sorted(district_ids)),
        status="published" if summary.published else summary.status,
        predictions=predictions,
        summary_status=summary.status,
        job_counts=summary.job_counts,
        artifact_dir=str(paths["artifact_path"].parent),
        warnings=list(summary.warnings),
    )


def _assemble_regional_with_fallback(
    *,
    mapping: RegionalMapping,
    national_predictions: dict[tuple[int, str, str, str], dict[str, str]],
    subregion_runs: dict[str, SubregionRun],
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """Combine regional predictions with national fallback, one per national key."""
    subregion_of = {item.district_id: item.subregion for item in mapping.districts}
    combined: list[dict[str, Any]] = []
    scope_counts: dict[str, int] = {}
    fallback_log: list[dict[str, Any]] = []
    logged_fallbacks: set[tuple[str, int]] = set()

    for key in sorted(national_predictions):
        horizon, fold, district_id, target_week = key
        if horizon not in horizons:
            continue
        subregion = subregion_of.get(district_id, "UNKNOWN")
        run = subregion_runs.get(subregion)
        regional_row = run.predictions.get(key) if run else None
        single_district = mapping.is_single_district_subregion(subregion)
        if single_district:
            scope_source = "national_fallback_single_district"
            chosen = national_predictions[key]
        elif regional_row is not None:
            scope_source = "dhis2_subregion_pooled"
            chosen = regional_row
        else:
            scope_source = "national_fallback_no_regional_prediction"
            chosen = national_predictions[key]
            marker = (subregion, horizon)
            if marker not in logged_fallbacks:
                logged_fallbacks.add(marker)
                fallback_log.append(
                    {
                        "subregion": subregion,
                        "horizon": horizon,
                        "reason": "no valid regional candidate produced predictions for this fold",
                        "subregion_summary_status": run.summary_status if run else "not_run",
                    }
                )
        scope_counts[scope_source] = scope_counts.get(scope_source, 0) + 1
        combined.append({**chosen, "subregion": subregion, "scope_source": scope_source})
    for subregion in mapping.fallback_subregions():
        fallback_log.append(
            {
                "subregion": subregion,
                "horizon": "all",
                "reason": "single-district subregion is served by the national candidate",
                "scope_source": "national_fallback_single_district",
            }
        )
    return combined, scope_counts, fallback_log


# --------------------------------------------------------------------------- #
# Metrics on combined predictions (reusing shared evaluators)
# --------------------------------------------------------------------------- #
def _record_from_row(row: dict[str, Any]) -> PredictionRecord:
    return PredictionRecord(
        district_id=str(row["district_id"]),
        horizon=int(row["horizon"]),
        origin_date=date.fromisoformat(str(row["forecast_origin_week"])),
        target_date=date.fromisoformat(str(row["target_week"])),
        actual=Decimal(str(row["observed_cases"])),
        predicted=Decimal(str(row["predicted_cases"])),
        threshold=Decimal(str(row["district_surge_threshold"])),
        actual_surge=str(row["observed_surge_status"]).strip().lower() in {"1", "true", "yes"},
        predicted_surge=str(row["count_threshold_surge_classification"]).strip().lower()
        in {"1", "true", "yes"},
        surge_score=Decimal(str(row["predicted_surge_score"])),
    )


def _horizon_metrics(rows: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        records = [_record_from_row(row) for row in rows if int(row["horizon"]) == horizon]
        if not records:
            metrics[horizon] = {"count": {"n": 0}, "surge_count_threshold": {"n": 0}}
            continue
        count_metrics, _ = evaluate_predictions(records)
        surge_metrics = classification_metrics(
            [record.actual_surge for record in records],
            [record.predicted_surge for record in records],
            [record.surge_score for record in records],
        )
        metrics[horizon] = {"count": count_metrics, "surge_count_threshold": surge_metrics}
    return metrics


def _count_only(records: list[PredictionRecord]) -> dict[str, Any]:
    count_metrics, _ = evaluate_predictions(records)
    return count_metrics


def _subregion_metric_rows(
    *,
    model_name: str,
    strategy: str,
    combined: list[dict[str, Any]],
    mapping: RegionalMapping,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for subregion in mapping.subregions:
        for horizon in horizons:
            rows = [
                row
                for row in combined
                if row["subregion"] == subregion and int(row["horizon"]) == horizon
            ]
            if not rows:
                continue
            records = [_record_from_row(row) for row in rows]
            count_metrics, _ = evaluate_predictions(records)
            surge = classification_metrics(
                [r.actual_surge for r in records],
                [r.predicted_surge for r in records],
                [r.surge_score for r in records],
            )
            scope_sources = sorted({row["scope_source"] for row in rows})
            out.append(
                {
                    "model_name": model_name,
                    "strategy": strategy,
                    "subregion": subregion,
                    "horizon": horizon,
                    "district_count": mapping.subregion_counts[subregion],
                    "scope_source": "|".join(scope_sources),
                    "n": count_metrics["n"],
                    "mae": count_metrics["mae"],
                    "rmse": count_metrics["rmse"],
                    "mape": count_metrics["mape"],
                    "wape": count_metrics["wape"],
                    "r_squared": count_metrics["r_squared"],
                    "surge_n": surge["n"],
                    "tp": surge["tp"],
                    "tn": surge["tn"],
                    "fp": surge["fp"],
                    "fn": surge["fn"],
                    "sensitivity": surge["sensitivity"],
                    "specificity": surge["specificity"],
                    "precision": surge["precision"],
                    "f1": surge["f1"],
                    "f2": surge["f2"],
                    "pr_auc": surge["pr_auc"],
                }
            )
    return out


def _district_metric_rows(
    *,
    model_name: str,
    strategy: str,
    combined: list[dict[str, Any]],
    mapping: RegionalMapping,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    names = {item.district_id: item.district_name for item in mapping.districts}
    subregions = {item.district_id: item.subregion for item in mapping.districts}
    out: list[dict[str, Any]] = []
    districts = sorted({str(row["district_id"]) for row in combined})
    for district_id in districts:
        for horizon in horizons:
            rows = [
                row
                for row in combined
                if str(row["district_id"]) == district_id and int(row["horizon"]) == horizon
            ]
            if not rows:
                continue
            records = [_record_from_row(row) for row in rows]
            count_metrics, _ = evaluate_predictions(records)
            out.append(
                {
                    "model_name": model_name,
                    "strategy": strategy,
                    "district_id": district_id,
                    "district_name": names.get(district_id, district_id),
                    "subregion": subregions.get(district_id, "UNKNOWN"),
                    "horizon": horizon,
                    "scope_source": "|".join(sorted({row["scope_source"] for row in rows})),
                    "n": count_metrics["n"],
                    "mae": count_metrics["mae"],
                    "rmse": count_metrics["rmse"],
                    "mape": count_metrics["mape"],
                    "wape": count_metrics["wape"],
                    "r_squared": count_metrics["r_squared"],
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Atomic publication
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stage_and_publish(writers: list[tuple[Path, Callable[[Path], None]]]) -> None:
    staged: dict[Path, Path] = {}
    try:
        for target, writer in writers:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.", suffix=".staged", dir=target.parent, delete=False
            ) as handle:
                staged_path = Path(handle.name)
            writer(staged_path)
            staged[staged_path] = target
        with publish_staged_outputs(staged):
            pass
    except BaseException:
        for staged_path in staged:
            staged_path.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# Experiment manifest
# --------------------------------------------------------------------------- #
def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "scipy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[name] = "not_installed"
    return versions


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
@dataclass
class RegionalValidationSummary:
    model_name: str
    model_key: str
    status: str
    dry_run: bool
    horizons: tuple[int, ...]
    max_folds: int
    poolable_subregions: list[str] = field(default_factory=list)
    fallback_subregions: list[str] = field(default_factory=list)
    national_baseline_source: str = ""
    scope_source_counts: dict[str, int] = field(default_factory=dict)
    combined_metrics_by_horizon: dict[int, dict[str, Any]] = field(default_factory=dict)
    fallback_log: list[dict[str, Any]] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    subregion_status: dict[str, str] = field(default_factory=dict)
    published: bool = False
    warnings: list[str] = field(default_factory=list)
    generated_at_utc: str = ""


def run_regional_pooling_experiment(
    config: PooledTreeConfig,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    max_folds: int = 12,
    dry_run: bool = False,
    replace: bool = False,
    dataset_path: Path = DATASET_PATH,
    mapping_path: Path = REGIONAL_MAPPING_PATH,
    root: Path = EXPERIMENT_ROOT,
    rows: list[dict[str, Any]] | None = None,
    expected_subregion_counts: dict[str, int] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RegionalValidationSummary:
    """Run the DHIS2-subregion-pooled experiment for one tree model family.

    ``expected_subregion_counts`` defaults to the reviewed audit and should be
    left at its default in production; it is overridable only for unit tests on
    a small synthetic mapping.
    """
    unsupported = sorted(set(horizons) - set(HORIZONS))
    if unsupported:
        raise RegionalTreeValidationError("Unsupported horizon(s): " + ", ".join(map(str, unsupported)))
    if max_folds <= 0:
        raise RegionalTreeValidationError("max_folds must be greater than zero.")

    # Inherit the production thread count unless explicitly overridden.
    if REGIONAL_CONTROLLED_THREADS is not None:
        config = dataclass_replace(config, controlled_threads=REGIONAL_CONTROLLED_THREADS)

    rows, national_folds = _load_dataset_and_national_folds(dataset_path, horizons, max_folds, rows=rows)
    approved_ids = {str(row["district_id"]) for row in rows}
    mapping = load_regional_mapping(
        mapping_path,
        approved_district_ids=approved_ids,
        expected_subregion_counts=expected_subregion_counts,
    )
    feasibility = build_feasibility_audit(mapping, national_folds, horizons)

    summary = RegionalValidationSummary(
        model_name=config.model_name,
        model_key=config.model_key,
        status="dry-run" if dry_run else "running",
        dry_run=dry_run,
        horizons=tuple(horizons),
        max_folds=max_folds,
        poolable_subregions=list(mapping.poolable_subregions()),
        fallback_subregions=list(mapping.fallback_subregions()),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    paths = experiment_output_paths(config.model_key, root=root)
    summary.artifact_paths = {name: str(path) for name, path in paths.items()}
    summary.artifact_paths["region_mapping_csv"] = str(root / "region_mapping.csv")
    summary.artifact_paths["feasibility_audit_json"] = str(root / "feasibility_audit.json")

    if dry_run:
        # Validate mappings, dependencies, folds and destinations without fitting.
        import importlib.util

        if importlib.util.find_spec(config.dependency) is None:
            summary.warnings.append(
                f"{config.dependency} is not installed; install requirements-models.txt before a real run."
            )
        empty_folds = [horizon for horizon in horizons if not national_folds.get(horizon)]
        if empty_folds:
            summary.warnings.append(
                "No validation folds available for horizon(s): "
                + ", ".join(str(h) for h in empty_folds)
            )
        summary.status = "dry-run"
        summary.scope_source_counts = {
            "regional_candidate_subregions": len(mapping.poolable_subregions()),
            "national_fallback_subregions": len(mapping.fallback_subregions()),
        }
        summary.combined_metrics_by_horizon = {}
        summary.fallback_log = [
            row for row in feasibility["per_subregion_horizon_fold"] if row["decision"] != "regional_candidate"
        ][:50]
        summary.subregion_status = {name: "planned" for name in mapping.poolable_subregions()}
        return summary

    # 1) National baseline predictions (reuse-or-rebuild).
    baseline = _resolve_national_baseline(
        config,
        rows=rows,
        national_folds=national_folds,
        horizons=horizons,
        max_folds=max_folds,
        dataset_path=dataset_path,
        replace=replace,
        root=root,
        progress_callback=progress_callback,
    )
    summary.national_baseline_source = baseline["source"]
    national_predictions = baseline["predictions"]

    # 2) Per-subregion regional validation (Kampala excluded from regional pooling).
    subregion_runs: dict[str, SubregionRun] = {}
    for subregion in mapping.poolable_subregions():
        district_ids = frozenset(mapping.district_ids_in(subregion))
        run = _run_subregion(
            config,
            subregion=subregion,
            district_ids=district_ids,
            rows=rows,
            national_folds=national_folds,
            horizons=horizons,
            max_folds=max_folds,
            dataset_path=dataset_path,
            replace=replace,
            root=root,
            progress_callback=progress_callback,
        )
        subregion_runs[subregion] = run
        summary.subregion_status[subregion] = run.status
        summary.warnings.extend(f"[{subregion}] {warning}" for warning in run.warnings)

    # 3) Regional-with-fallback assembly.
    combined, scope_counts, fallback_log = _assemble_regional_with_fallback(
        mapping=mapping,
        national_predictions=national_predictions,
        subregion_runs=subregion_runs,
        horizons=horizons,
    )
    summary.scope_source_counts = scope_counts
    summary.fallback_log = fallback_log
    summary.combined_metrics_by_horizon = _horizon_metrics(combined, horizons)

    # 4) Feature importance (concatenated per subregion, tagged).
    importance_rows = _collect_feature_importance(config.model_key, mapping, root=root)

    # 5) Metric tables.
    subregion_metrics = _subregion_metric_rows(
        model_name=config.model_name,
        strategy="regional_with_fallback",
        combined=combined,
        mapping=mapping,
        horizons=horizons,
    )
    district_metrics = _district_metric_rows(
        model_name=config.model_name,
        strategy="regional_with_fallback",
        combined=combined,
        mapping=mapping,
        horizons=horizons,
    )

    # 6) Diagnostics, model card, top-level validation summary.
    diagnostics = {
        "model_name": config.model_name,
        "strategy": "regional_with_fallback",
        "national_baseline": {
            "source": baseline["source"],
            "predictions_path": baseline["predictions_path"],
            "reasons_for_rebuild": baseline["reasons"],
        },
        "dataset_signature": _dataset_signature(dataset_path),
        "mapping_signature": {"path": str(mapping.source_path), "sha256": mapping.source_sha256},
        "feasibility_audit": feasibility,
        "subregion_runs": {
            name: {
                "status": run.status,
                "summary_status": run.summary_status,
                "job_counts": run.job_counts,
                "district_ids": list(run.district_ids),
                "artifact_dir": run.artifact_dir,
            }
            for name, run in subregion_runs.items()
        },
        "scope_source_counts": scope_counts,
        "fallback_log": fallback_log,
        "package_versions": _package_versions(),
        "random_state": RANDOM_STATE,
        "controlled_threads": config.controlled_threads,
        "controlled_threads_note": (
            "Regional candidates inherit the model family's production controlled thread "
            "count, matching the national configuration exactly."
        ),
        "permutation_importance_computed": REGIONAL_COMPUTE_PERMUTATION_IMPORTANCE,
        "generated_at_utc": summary.generated_at_utc,
    }
    model_card = _regional_model_card(config, mapping, summary, baseline, horizons)
    summary_payload = _regional_summary_payload(config, mapping, summary, baseline, combined, horizons)

    _stage_and_publish(
        [
            (paths["validation_summary"], lambda p: _write_json(p, summary_payload)),
            (paths["validation_predictions"], lambda p: _write_csv(p, combined, REGIONAL_PREDICTION_COLUMNS)),
            (paths["subregion_horizon_metrics"], lambda p: _write_csv(p, subregion_metrics, SUBREGION_METRIC_COLUMNS)),
            (paths["district_horizon_metrics"], lambda p: _write_csv(p, district_metrics, DISTRICT_METRIC_COLUMNS)),
            (paths["fit_diagnostics"], lambda p: _write_json(p, diagnostics)),
            (paths["model_card"], lambda p: _write_json(p, model_card)),
            (paths["feature_importance"], lambda p: _write_csv(p, importance_rows, _IMPORTANCE_COLUMNS)),
        ]
    )
    # Shared experiment artefacts (mapping + feasibility) are written atomically too.
    _stage_and_publish(
        [
            (root / "region_mapping.csv", lambda p: write_region_mapping_csv(mapping, p)),
            (root / "feasibility_audit.json", lambda p: _write_json(p, feasibility)),
        ]
    )
    summary.published = True
    summary.status = "success"
    return summary


_IMPORTANCE_COLUMNS = (
    "model_name",
    "subregion",
    "horizon",
    "validation_fold",
    "importance_method",
    "feature_name",
    "importance",
)


def _collect_feature_importance(model_key: str, mapping: RegionalMapping, *, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subregion in mapping.poolable_subregions():
        path = _subregion_artifact_paths(model_key, subregion, root=root)["feature_importance_path"]
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "model_name": row.get("model_name", ""),
                        "subregion": subregion,
                        "horizon": row.get("horizon", ""),
                        "validation_fold": row.get("validation_fold", ""),
                        "importance_method": row.get("importance_method", ""),
                        "feature_name": row.get("feature_name", ""),
                        "importance": row.get("importance", ""),
                    }
                )
    return rows


def _regional_model_card(config, mapping, summary, baseline, horizons) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "experiment": "dhis2_subregion_pooling",
        "strategy": "regional_with_fallback",
        "model_scope": "dhis2_subregion_pooled_with_national_fallback",
        "horizons_weeks": list(horizons),
        "subregion_pooling": (
            "One count regressor and one class-balanced surge classifier are fitted per "
            "DHIS2 subregion, horizon and fold, using only that subregion's districts. "
            "Kampala is a single district and always uses the national candidate "
            "(scope_source=national_fallback_single_district)."
        ),
        "national_fallback": (
            "Where a subregion/horizon/fold produced no valid regional candidate, the "
            "matched national prediction is used and the reason is recorded. National and "
            "regional forecasts are never averaged."
        ),
        "national_baseline_source": baseline["source"],
        "validation": (
            "The shared national rolling-origin folds are scoped to each subregion, so "
            "national and regional candidates use identical origin and target dates. "
            "Preprocessing, district encoding and surge thresholds are fitted on scoped "
            "training rows only; outcomes are never imputed."
        ),
        "poolable_subregions": list(mapping.poolable_subregions()),
        "national_fallback_subregions": list(mapping.fallback_subregions()),
        "not_for_production": (
            "Experimental only. This does not replace production models, forecasts or "
            "the model-selection artefact, and no model is promoted from this run."
        ),
        "limitations": [
            "Regional pooling reduces training data per model; sparse subregions may be unstable.",
            "Subregion-level and district-level metrics rest on few validation predictions each.",
            "Relative WAPE reduction is an error metric, not accuracy.",
        ],
    }


def _regional_summary_payload(config, mapping, summary, baseline, combined, horizons) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "model_key": config.model_key,
        "experiment": "dhis2_subregion_pooling",
        "strategy": "regional_with_fallback",
        "status": "success",
        "published": True,
        "horizons": list(horizons),
        "max_folds": summary.max_folds,
        "validation_method": "rolling_origin",
        "no_random_shuffle": True,
        "national_baseline_source": baseline["source"],
        "poolable_subregions": list(mapping.poolable_subregions()),
        "national_fallback_subregions": list(mapping.fallback_subregions()),
        "scope_source_counts": summary.scope_source_counts,
        "combined_metrics_by_horizon": summary.combined_metrics_by_horizon,
        "fallback_log": summary.fallback_log,
        "prediction_rows": len(combined),
        "mapping_signature": {"path": str(mapping.source_path), "sha256": mapping.source_sha256},
        "generated_at_utc": summary.generated_at_utc,
        "relative_wape_reduction_label": (
            "Relative error reduction against the national baseline; never percentage accuracy."
        ),
    }


# --------------------------------------------------------------------------- #
# Matched comparison + district-cluster bootstrap
# --------------------------------------------------------------------------- #
def _pooled_wape(records: list[PredictionRecord]) -> Decimal | None:
    total_error = sum((abs(r.predicted - r.actual) for r in records), Decimal("0"))
    total_actual = sum((r.actual for r in records), Decimal("0"))
    if total_actual == 0:
        return None
    return total_error / total_actual * Decimal("100")


def _relative_reduction(national: Decimal | None, regional: Decimal | None) -> float | None:
    if national is None or regional is None or national == 0:
        return None
    return float((national - regional) / national * Decimal("100"))


def _bootstrap_paired_reduction(
    district_records: dict[str, dict[str, list[PredictionRecord]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """District-cluster bootstrap of the paired pooled-WAPE difference.

    Resamples complete district histories with replacement. Deterministic.
    """
    districts = sorted(district_records)
    if len(districts) < 2:
        return {"iterations": 0, "note": "too few districts for a bootstrap interval"}
    rng = random.Random(seed)
    diffs: list[float] = []
    rels: list[float] = []
    for _ in range(iterations):
        sample = [districts[rng.randrange(len(districts))] for _ in districts]
        nat: list[PredictionRecord] = []
        reg: list[PredictionRecord] = []
        for district in sample:
            nat.extend(district_records[district]["national"])
            reg.extend(district_records[district]["regional"])
        nat_wape = _pooled_wape(nat)
        reg_wape = _pooled_wape(reg)
        if nat_wape is None or reg_wape is None:
            continue
        diffs.append(float(nat_wape - reg_wape))
        rel = _relative_reduction(nat_wape, reg_wape)
        if rel is not None:
            rels.append(rel)
    if not diffs:
        return {"iterations": 0, "note": "bootstrap produced no usable resamples"}
    diffs.sort()
    rels.sort()

    def _pct(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    return {
        "iterations": len(diffs),
        "seed": seed,
        "wape_point_difference_national_minus_regional": {
            "median": _pct(diffs, 0.5),
            "ci95_low": _pct(diffs, 0.025),
            "ci95_high": _pct(diffs, 0.975),
        },
        "relative_wape_reduction_pct": {
            "median": _pct(rels, 0.5) if rels else None,
            "ci95_low": _pct(rels, 0.025) if rels else None,
            "ci95_high": _pct(rels, 0.975) if rels else None,
        },
        "interpretation": (
            "Positive values favour regional pooling (lower WAPE). A 95% interval that "
            "includes zero is not evidence of improvement."
        ),
    }


def run_regional_comparison(
    config: PooledTreeConfig,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    dataset_path: Path = DATASET_PATH,
    mapping_path: Path = REGIONAL_MAPPING_PATH,
    root: Path = EXPERIMENT_ROOT,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    expected_subregion_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Match national vs regional-with-fallback predictions and quantify difference."""
    mapping = load_regional_mapping(mapping_path, expected_subregion_counts=expected_subregion_counts)
    regional_paths = experiment_output_paths(config.model_key, root=root)
    regional_predictions_path = regional_paths["validation_predictions"]
    if not regional_predictions_path.exists():
        raise RegionalTreeValidationError(
            f"Regional predictions are missing for {config.model_name}; run the experiment first."
        )
    # National predictions: prefer the fresh experimental baseline if present, else production.
    fresh_national = _national_baseline_paths(config.model_key, root=root)["predictions_path"]
    national_predictions_path = (
        fresh_national if fresh_national.exists() else _existing_national_paths(config.model_key)["predictions"]
    )
    if not national_predictions_path.exists():
        raise RegionalTreeValidationError(f"National predictions are missing for {config.model_name}.")

    national = _read_predictions(national_predictions_path, horizons)
    regional = _read_predictions(regional_predictions_path, horizons)

    national_keys = set(national)
    regional_keys = set(regional)
    missing_in_regional = national_keys - regional_keys
    missing_in_national = regional_keys - national_keys
    complete = not missing_in_regional and not missing_in_national

    names = {item.district_id: item.district_name for item in mapping.districts}
    subregions = {item.district_id: item.subregion for item in mapping.districts}

    comparison_rows: list[dict[str, Any]] = []
    district_records: dict[int, dict[str, dict[str, list[PredictionRecord]]]] = {
        horizon: {} for horizon in horizons
    }
    matched_keys = sorted(national_keys & regional_keys)
    for key in matched_keys:
        horizon, _fold, district_id, _target = key
        nat_row = national[key]
        reg_row = regional[key]
        district_records[horizon].setdefault(
            district_id, {"national": [], "regional": []}
        )
        district_records[horizon][district_id]["national"].append(_record_from_row(nat_row))
        district_records[horizon][district_id]["regional"].append(_record_from_row(reg_row))

    per_horizon: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        improved = unchanged = worsened = 0
        for district_id, sides in sorted(district_records[horizon].items()):
            nat_wape = _pooled_wape(sides["national"])
            reg_wape = _pooled_wape(sides["regional"])
            rel = _relative_reduction(nat_wape, reg_wape)
            nat_count = _count_only(sides["national"])
            reg_count = _count_only(sides["regional"])
            scope_sources = sorted(
                {regional[k]["scope_source"] for k in matched_keys if k[0] == horizon and k[2] == district_id}
            )
            if nat_wape is None or reg_wape is None:
                outcome = "undefined"
            elif reg_wape < nat_wape:
                outcome = "improved"
                improved += 1
            elif reg_wape > nat_wape:
                outcome = "worsened"
                worsened += 1
            else:
                outcome = "unchanged"
                unchanged += 1
            comparison_rows.append(
                {
                    "model_name": config.model_name,
                    "horizon": horizon,
                    "district_id": district_id,
                    "district_name": names.get(district_id, district_id),
                    "subregion": subregions.get(district_id, "UNKNOWN"),
                    "scope_source": "|".join(scope_sources),
                    "n": nat_count["n"],
                    "national_wape": nat_count["wape"],
                    "regional_wape": reg_count["wape"],
                    "relative_wape_reduction_pct": (round(rel, 6) if rel is not None else None),
                    "national_mae": nat_count["mae"],
                    "regional_mae": reg_count["mae"],
                    "district_outcome": outcome,
                }
            )
        national_records = [r for sides in district_records[horizon].values() for r in sides["national"]]
        regional_records = [r for sides in district_records[horizon].values() for r in sides["regional"]]
        nat_pooled = _pooled_wape(national_records)
        reg_pooled = _pooled_wape(regional_records)
        bootstrap = _bootstrap_paired_reduction(
            district_records[horizon], iterations=bootstrap_iterations, seed=BOOTSTRAP_SEED + horizon
        )
        per_horizon[horizon] = {
            "matched_predictions": len(national_records),
            "national_pooled_wape": str(nat_pooled) if nat_pooled is not None else None,
            "regional_pooled_wape": str(reg_pooled) if reg_pooled is not None else None,
            "relative_wape_reduction_pct": _relative_reduction(nat_pooled, reg_pooled),
            "districts_improved": improved,
            "districts_unchanged": unchanged,
            "districts_worsened": worsened,
            "districts_total": improved + unchanged + worsened,
            "district_cluster_bootstrap": bootstrap,
        }

    summary_payload = {
        "model_name": config.model_name,
        "model_key": config.model_key,
        "experiment": "dhis2_subregion_pooling",
        "national_predictions_path": str(national_predictions_path),
        "regional_predictions_path": str(regional_predictions_path),
        "match_keys": "model_family, horizon, validation_fold, district_id, target_week",
        "comparison_complete": complete,
        "unmatched_national_keys": len(missing_in_regional),
        "unmatched_regional_keys": len(missing_in_national),
        "count_selection_rule": "lowest WAPE where available, with RMSE fallback",
        "surge_selection_rule": (
            "highest F2, then sensitivity, PR-AUC, precision, specificity; accuracy never selects"
        ),
        "relative_wape_reduction_label": (
            "Relative error reduction against the national baseline; never percentage accuracy."
        ),
        "per_horizon": per_horizon,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not complete:
        summary_payload["comparison_incomplete_warning"] = (
            "Prediction keys did not match exactly; improvement is not claimed."
        )

    _stage_and_publish(
        [
            (root / f"comparison_summary_{config.model_key}.json", lambda p: _write_json(p, summary_payload)),
            (root / f"comparison_rows_{config.model_key}.csv", lambda p: _write_csv(p, comparison_rows, COMPARISON_ROW_COLUMNS)),
        ]
    )
    return summary_payload
