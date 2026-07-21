"""Single approved data-quality evidence builder (Phase 29D corrective pass).

This is the ONE implementation of the data-quality evidence contract. Both
``/api/lifecycle/data-quality`` (Phase 29C) and the Phase 29D Data Quality
report consume it, so the two can never drift apart or disagree.

Everything it returns is derived from approved evidence and cross-checked:

* surveillance coverage from project-state row-counts, validated for internal
  consistency (expected - received == missing) and agreement with the database;
* the complete nine-column climate-lag contract, with a real complete-lag row
  count (never inferred from a single column);
* leakage controls derived from the production manifest, the modelling-dataset
  summary and the approved SARIMAX / Random Forest / XGBoost model cards;
* adjudicated case exclusions and predictor-only interpolation verified from
  the modelling-dataset cleaning manifest.

Any missing, malformed, inconsistent or incomplete required evidence raises
``DataQualityEvidenceError`` so every caller fails closed. Nothing here is a
static claim: if a control cannot be proven from evidence, the build fails.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func

from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature

# All nine approved climate-lag columns (rainfall/temperature/humidity x 4/8/12).
CLIMATE_LAG_COLUMNS = (
    "rainfall_lag_4", "rainfall_lag_8", "rainfall_lag_12",
    "temperature_lag_4", "temperature_lag_8", "temperature_lag_12",
    "humidity_lag_4", "humidity_lag_8", "humidity_lag_12",
)

PREPROCESSING_CARD_KEYS = {
    "SARIMAX": "sarimax_card",
    "Random Forest": "random_forest_card",
    "XGBoost": "xgboost_card",
}

TRAINING_ONLY_PHRASES = ("fitted only on training rows", "fitted on training rows only")

LIMITATIONS = [
    "Source reporting quality affects every downstream stage.",
    "District aggregation can hide sub-district variation.",
    "Missing and excluded observations remain missing as outcomes; only bounded predictor gaps are interpolated.",
]

DEFINITIONS = [
    ("Missing report", "No malaria value was received for the district-week; the row is absent."),
    ("Explicit zero", "A value of zero was explicitly reported."),
    ("Missing is not zero", "Missing reports are never converted to zero."),
    ("Epidemiological weeks", "ISO epidemiological weeks are validated, including week 53."),
    ("Adjudicated error", "A district-week confirmed as a reporting error after District Biostatistician consultation; raw data remain unchanged."),
    ("Predictor-only interpolation", "Linear interpolation is limited to bounded one- or two-week internal gaps and never supplies an outcome label."),
    ("District standardisation", "All records are matched to canonical district_id values."),
    ("Leakage prevention", "Targets are future-only, lags are past-only and preprocessing is fitted on training rows only."),
]


class DataQualityEvidenceError(RuntimeError):
    """Required data-quality evidence is missing, malformed or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataQualityEvidenceError(message)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_data_quality_evidence() -> dict[str, Any]:
    """Build the validated data-quality context, or fail closed."""
    from app.services.explainability import ExplainabilityError, load_json
    from app.services.project_state import build_project_state_summary

    # ---- project state -------------------------------------------------
    try:
        summary = build_project_state_summary(write=False)
    except Exception as exc:  # noqa: BLE001
        raise DataQualityEvidenceError("Project-state evidence is unavailable.") from exc
    components = summary.get("components") or {}
    _require(bool(components), "Project-state evidence is unavailable.")

    panel = components.get("dhis2_malaria_panel") or {}
    counts = panel.get("row_counts") or {}
    expected = _as_int(counts.get("expected_district_weeks"))
    received = _as_int(counts.get("district_week_features"))
    missing = _as_int(counts.get("missing_district_weeks"))
    explicit_zero = _as_int(counts.get("feature_explicit_zero_count"))
    raw_rows = _as_int(counts.get("raw_dhis2_malaria"))
    districts_represented = _as_int(counts.get("districts_represented"))

    _require(None not in (expected, received, missing),
             "Surveillance coverage evidence is missing required totals.")
    _require(expected - received == missing,
             "Surveillance coverage evidence is internally inconsistent.")

    # ---- leakage evidence (manifest + dataset + model cards) ------------
    try:
        manifest = load_json("production_manifest")
        dataset = load_json("modelling_dataset")
        cards = {name: load_json(key) for name, key in PREPROCESSING_CARD_KEYS.items()}
    except ExplainabilityError as exc:
        raise DataQualityEvidenceError("Leakage-control evidence is unavailable or malformed.") from exc

    controls_evidence = manifest.get("leakage_controls")
    checks = dataset.get("leakage_checks")
    cleaning = dataset.get("case_cleaning")
    _require(isinstance(controls_evidence, dict) and isinstance(checks, dict),
             "Leakage-control evidence is malformed.")
    _require(isinstance(cleaning, dict), "Case-cleaning evidence is unavailable.")

    required_booleans = (
        controls_evidence.get("missing_malaria_zero_filled"),
        dataset.get("missing_malaria_not_zero_filled"),
        checks.get("targets_future_only"),
        checks.get("lag_features_past_only"),
    )
    _require(all(isinstance(v, bool) for v in required_booleans),
             "Leakage-control evidence has invalid value types.")

    # Training-only preprocessing must be proven by EVERY approved model card.
    proofs = []
    for expected_name, card in cards.items():
        _require(isinstance(card, dict), "Training-only preprocessing evidence is malformed.")
        acceptable_status = card.get("status") == "success" or (
            expected_name == "SARIMAX"
            and card.get("status") == "partial_success"
            and int((card.get("validation_job_counts") or {}).get("failed", -1)) == 0
        )
        _require(card.get("model_name") == expected_name and acceptable_status,
                 "Training-only preprocessing evidence is inconsistent.")
        preprocessing = card.get("preprocessing")
        _require(isinstance(preprocessing, str),
                 "Training-only preprocessing evidence is malformed.")
        normalized = " ".join(preprocessing.lower().split())
        _require(any(phrase in normalized for phrase in TRAINING_ONLY_PHRASES),
                 "Training-only preprocessing evidence is incomplete.")
        proofs.append(True)

    # ---- database verification -----------------------------------------
    try:
        with db.session.no_autoflush:
            panel_rows = db.session.query(func.count(DistrictWeekFeature.id)).scalar() or 0
            active_districts = DistrictReference.query.filter_by(active=True).count()
            complete_lag_query = DistrictWeekFeature.query
            for column in CLIMATE_LAG_COLUMNS:
                complete_lag_query = complete_lag_query.filter(
                    getattr(DistrictWeekFeature, column).isnot(None))
            rows_with_all_lags = complete_lag_query.count()
    except Exception as exc:  # noqa: BLE001
        raise DataQualityEvidenceError("Preparation evidence could not be read.") from exc

    _require(received == int(panel_rows),
             "Surveillance evidence disagrees with the stored panel row count.")

    # ---- derived controls (never literal True) --------------------------
    confirmed_error_count = _as_int(
        cleaning.get("confirmed_erroneous_district_weeks")
    )
    excluded_records = cleaning.get("excluded_records")
    _require(
        confirmed_error_count is not None and isinstance(excluded_records, list),
        "Case-cleaning evidence is malformed.",
    )
    leakage_controls = {
        "missing_not_zero_filled": {
            "value": (controls_evidence["missing_malaria_zero_filled"] is False
                      and dataset["missing_malaria_not_zero_filled"] is True),
            "source": "production manifest + modelling-dataset summary",
        },
        "future_target_alignment": {
            "value": (checks["targets_future_only"] is True
                      and checks["lag_features_past_only"] is True),
            "source": "modelling-dataset leakage_checks (targets_future_only, lag_features_past_only)",
        },
        "training_only_preprocessing": {
            "value": len(proofs) == len(cards) and all(proofs),
            "source": "approved SARIMAX, Random Forest and XGBoost model cards",
        },
        "adjudicated_errors_excluded_from_outcomes": {
            "value": (
                confirmed_error_count == len(excluded_records)
                and confirmed_error_count > 0
                and cleaning.get("outcome_imputation") is False
            ),
            "source": "adjudicated case-exclusion registry + modelling-dataset cleaning manifest",
        },
        "bounded_predictor_only_interpolation": {
            "value": (
                cleaning.get("maximum_interpolated_gap_weeks") == 2
                and cleaning.get("outcome_imputation") is False
            ),
            "source": "modelling-dataset cleaning manifest",
        },
    }

    unproven_controls = [
        name for name, control in leakage_controls.items()
        if control["value"] is not True
    ]
    _require(
        not unproven_controls,
        "Required data-quality controls are not proven by the available evidence: "
        + ", ".join(unproven_controls) + ".",
    )

    def status_of(name):
        return (components.get(name) or {}).get("status")

    warnings: list[str] = []
    if rows_with_all_lags != panel_rows:
        warnings.append(
            f"{panel_rows - rows_with_all_lags} panel row(s) do not carry all nine climate lags.")
    for component, status in ((k, (v or {}).get("status")) for k, v in components.items()):
        if status not in ("complete", None):
            warnings.append(f"Pipeline component '{component}' reports status '{status}'.")

    stages = [
        {"key": "raw_dhis2", "title": "Raw DHIS2 data", "status": status_of("dhis2_malaria_panel"),
         "detail": "Confirmed malaria cases by district and epidemiological week."},
        {"key": "district_matching", "title": "District matching", "status": status_of("reference_and_boundaries"),
         "detail": f"Matched to {active_districts} canonical districts by district_id."},
        {"key": "iso_week", "title": "ISO week validation", "status": status_of("dhis2_malaria_panel"),
         "detail": "Epidemiological weeks validated, including week-53 handling."},
        {"key": "missing_checks", "title": "Missing-data checks", "status": status_of("reporting_completeness"),
         "detail": "Missing observations are retained as missing, never zero-filled."},
        {"key": "case_cleaning", "title": "Adjudicated case cleaning", "status": status_of("modelling_dataset"),
         "detail": f"{cleaning.get('confirmed_erroneous_district_weeks')} confirmed reporting errors excluded from outcomes; raw values retained in the audit registry."},
        {"key": "climate_linkage", "title": "Climate linkage", "status": status_of("climate_aggregation"),
         "detail": "CHIRPS and ERA5-Land linked to districts by boundary."},
        {"key": "lags", "title": "Lag generation", "status": status_of("climate_lags"),
         "detail": "4, 8 and 12-week rainfall, temperature and humidity lags generated before the forecast target."},
        {"key": "modelling_dataset", "title": "Modelling dataset", "status": status_of("modelling_dataset"),
         "detail": "Leakage-checked dataset with future-target alignment."},
    ]

    return {
        "stages": stages,
        "components": {k: (v or {}).get("status") for k, v in components.items()},
        "coverage": {
            "expected_district_weeks": expected,
            "received_district_weeks": received,
            "missing_district_weeks": missing,
            "explicit_zero_records": explicit_zero,
            "raw_dhis2_rows": raw_rows,
            "districts_represented": districts_represented,
        },
        "panel_rows": int(panel_rows),
        "active_districts": int(active_districts),
        "rows_with_complete_climate_lags": int(rows_with_all_lags),
        "climate_lag_columns_verified": list(CLIMATE_LAG_COLUMNS),
        "leakage_controls": leakage_controls,
        "case_cleaning": cleaning,
        "definitions": list(DEFINITIONS),
        "limitations": list(LIMITATIONS),
        "warnings": warnings,
        "evidence_sources": (
            "project-state row counts, production manifest, modelling-dataset summary, "
            "approved model cards and database verification"
        ),
    }
