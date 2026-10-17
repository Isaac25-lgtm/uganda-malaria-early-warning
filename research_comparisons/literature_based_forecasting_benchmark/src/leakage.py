"""Programmatic leakage and output-isolation guards.

Every prediction carries provenance; every provenance record is checked for
future-information leakage. Any output path outside runs/<RUN_ID>/ is rejected.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from research_comparisons.literature_based_forecasting_benchmark import config as C

PROVENANCE_COLUMNS = (
    "run_id",
    "method_id",
    "adaptation_class",
    "district_id",
    "horizon",
    "fold_number",
    "origin_date",
    "target_date",
    "observed_cases",
    "predicted_cases",
    "max_case_source_date",
    "max_climate_source_date",
    "max_feature_source_date",
    "feature_cutoff_date",
    "training_max_target_date",
    "threshold",
    "threshold_history_max_date",
    "leakage_check_passed",
    "prediction_status",
    "failure_reason",
)


class OutputIsolationError(RuntimeError):
    """Raised when an output path escapes the run directory."""


class LeakageError(RuntimeError):
    """Raised when a prediction would use future information."""


def assert_output_isolated(path: Path) -> Path:
    """Resolve ``path`` and require it to live under runs/<RUN_ID>/ only."""
    resolved = Path(path).resolve()
    run_dir = C.RUN_DIR.resolve()
    under_run = resolved == run_dir or run_dir in resolved.parents
    if not under_run:
        raise OutputIsolationError(f"Output path is not under the run directory: {resolved}")
    for root in C.FORBIDDEN_OUTPUT_ROOTS:
        forbidden = (C.REPO_ROOT / root).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise OutputIsolationError(f"Output path is under a forbidden root {root!r}: {resolved}")
    return resolved


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def check_prediction_leakage(
    *,
    origin_date,
    target_date,
    feature_cutoff_date,
    training_max_target_date,
    threshold_history_max_date,
) -> tuple[bool, str]:
    """Return (passed, reason). Enforces origin-observable, training-only rules."""
    origin = _as_date(origin_date)
    problems: list[str] = []
    if _as_date(target_date) <= origin:
        problems.append("target_date not after origin_date")
    # Origin-week inputs are excluded by this benchmark's availability rule.
    # A prediction feature must therefore be sourced strictly before origin.
    if feature_cutoff_date is not None and _as_date(feature_cutoff_date) >= origin:
        problems.append("feature_cutoff_date is not strictly before origin_date")
    if training_max_target_date is not None and _as_date(training_max_target_date) > origin:
        problems.append("a training target was not known by the origin")
    if threshold_history_max_date is not None and _as_date(threshold_history_max_date) > origin:
        problems.append("threshold reference history extends past the origin")
    return (not problems, "; ".join(problems))


def training_max_target_date(trainable_rows, horizon: int):
    """Latest training target-week date (must be <= each fold's origin)."""
    dates = [
        _as_date(row[f"target_week_start_date_h{horizon}"])
        for row in trainable_rows
        if row.get(f"target_week_start_date_h{horizon}") not in (None, "")
    ]
    return max(dates) if dates else None


def threshold_history_max_date(train_rows):
    """Latest week used as threshold/history reference (must be <= origin)."""
    dates = [
        _as_date(row["week_start_date"])
        for row in train_rows
        if row.get("week_start_date") not in (None, "")
    ]
    return max(dates) if dates else None
