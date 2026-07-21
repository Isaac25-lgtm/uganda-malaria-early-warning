"""Core research data contract models.

These tables define storage only. They do not perform DHIS2 extraction,
preprocessing, climate ingestion, model training, forecast generation or
dashboard data wiring.
"""
from __future__ import annotations

from sqlalchemy.dialects.postgresql import JSONB

from app.extensions import db

APPROVED_MODEL_NAMES = ("Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost")
FORECAST_HORIZONS = (4, 8, 12)
RISK_LEVELS = ("Low", "Moderate", "High", "Very High", "No data")
VALIDATION_METHODS = ("rolling_origin", "forward_chaining")
MODEL_SCOPES = ("national_pooled", "regional_pooled", "district_specific", "other")
PIPELINE_RUN_TYPES = (
    "dhis2_extraction",
    "dhis2_reporting_completeness",
    "preprocessing",
    "climate_ingestion",
    "climate_aggregation",
    "climate_lag_creation",
    "feature_engineering",
    "model_training",
    "forecast_generation",
    "report_export",
)
PIPELINE_STATUSES = ("pending", "running", "success", "failed")


def _in_constraint(column_name: str, values: tuple[str, ...] | tuple[int, ...]) -> str:
    rendered = ", ".join(
        f"'{value}'" if isinstance(value, str) else str(value) for value in values
    )
    return f"{column_name} IN ({rendered})"


class DistrictReference(db.Model):
    """Canonical district lookup table used by all analytical joins."""

    __tablename__ = "district_reference"
    __table_args__ = (
        db.Index("ix_district_reference_district_id", "district_id"),
        db.Index("ix_district_reference_dhis2_orgunit_id", "dhis2_orgunit_id"),
        db.Index("ix_district_reference_ubos_code", "ubos_code"),
    )

    id = db.Column(db.Integer, primary_key=True)
    district_id = db.Column(db.String(64), nullable=False, unique=True)
    district_name = db.Column(db.String(160), nullable=False)
    dhis2_orgunit_id = db.Column(db.String(64), nullable=True)
    dhis2_orgunit_name = db.Column(db.String(160), nullable=True)
    ubos_code = db.Column(db.String(64), nullable=True)
    region = db.Column(db.String(120), nullable=True)
    subregion = db.Column(db.String(120), nullable=True)
    active = db.Column(
        db.Boolean, nullable=False, default=True, server_default=db.text("true")
    )
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=db.func.now(),
        onupdate=db.func.now(),
    )

    raw_dhis2_records = db.relationship(
        "RawDHIS2Malaria", back_populates="district", lazy="dynamic"
    )
    population_records = db.relationship(
        "UBOSPopulation", back_populates="district", lazy="dynamic"
    )
    week_features = db.relationship(
        "DistrictWeekFeature", back_populates="district", lazy="dynamic"
    )
    forecast_outputs = db.relationship(
        "ForecastOutput", back_populates="district", lazy="dynamic"
    )
    model_performance = db.relationship(
        "ModelPerformance", back_populates="district", lazy="dynamic"
    )


class UBOSPopulation(db.Model):
    """Approved district-year UBOS population denominators."""

    __tablename__ = "ubos_population"
    __table_args__ = (
        db.UniqueConstraint(
            "district_id", "year", name="uq_ubos_population_district_year"
        ),
        db.CheckConstraint("year >= 1900 AND year <= 2100", name="ck_ubos_population_year"),
        db.CheckConstraint("population > 0", name="ck_ubos_population_positive"),
        db.Index("ix_ubos_population_district_id", "district_id"),
        db.Index("ix_ubos_population_year", "year"),
    )

    id = db.Column(db.Integer, primary_key=True)
    district_id = db.Column(
        db.String(64), db.ForeignKey("district_reference.district_id"), nullable=False
    )
    district_name = db.Column(db.String(160), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    population = db.Column(db.Integer, nullable=False)
    source = db.Column(db.String(160), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=db.func.now(),
        onupdate=db.func.now(),
    )

    district = db.relationship("DistrictReference", back_populates="population_records")


class PipelineRun(db.Model):
    """Audit record for each major offline data/pipeline operation."""

    __tablename__ = "pipeline_runs"
    __table_args__ = (
        db.CheckConstraint(
            _in_constraint("run_type", PIPELINE_RUN_TYPES),
            name="ck_pipeline_runs_run_type",
        ),
        db.CheckConstraint(
            _in_constraint("status", PIPELINE_STATUSES),
            name="ck_pipeline_runs_status",
        ),
        db.CheckConstraint("row_count_raw IS NULL OR row_count_raw >= 0", name="ck_pipeline_runs_row_count_raw_nonnegative"),
        db.CheckConstraint(
            "row_count_processed IS NULL OR row_count_processed >= 0",
            name="ck_pipeline_runs_row_count_processed_nonnegative",
        ),
        db.CheckConstraint(
            "warning_count IS NULL OR warning_count >= 0",
            name="ck_pipeline_runs_warning_count_nonnegative",
        ),
        db.Index("ix_pipeline_runs_run_type", "run_type"),
        db.Index("ix_pipeline_runs_status", "status"),
        db.Index("ix_pipeline_runs_started_at", "started_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    run_type = db.Column(db.String(40), nullable=False)
    status = db.Column(
        db.String(20), nullable=False, default="pending", server_default="pending"
    )
    started_at = db.Column(db.DateTime(timezone=True), nullable=True)
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    source_name = db.Column(db.String(160), nullable=True)
    source_version_or_url = db.Column(db.String(500), nullable=True)
    parameters = db.Column(JSONB, nullable=True)
    row_count_raw = db.Column(db.Integer, nullable=True)
    row_count_processed = db.Column(db.Integer, nullable=True)
    warning_count = db.Column(db.Integer, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    manifest = db.Column(JSONB, nullable=True)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )

    creator = db.relationship("User", backref="pipeline_runs")
    raw_dhis2_records = db.relationship(
        "RawDHIS2Malaria", back_populates="pipeline_run", lazy="dynamic"
    )
    district_week_features = db.relationship(
        "DistrictWeekFeature", back_populates="source_pipeline_run", lazy="dynamic"
    )
    district_week_climate = db.relationship(
        "DistrictWeekClimate", back_populates="pipeline_run", lazy="dynamic"
    )
    forecast_outputs = db.relationship(
        "ForecastOutput", back_populates="pipeline_run", lazy="dynamic"
    )
    model_performance = db.relationship(
        "ModelPerformance", back_populates="pipeline_run", lazy="dynamic"
    )


class RawDHIS2Malaria(db.Model):
    """Raw weekly DHIS2 malaria analytics rows before cleaning."""

    __tablename__ = "raw_dhis2_malaria"
    __table_args__ = (
        db.UniqueConstraint(
            "pipeline_run_id",
            "dhis2_orgunit_id",
            "data_element_id",
            "period",
            name="uq_raw_dhis2_run_orgunit_dataelement_period",
        ),
        db.CheckConstraint("epi_week BETWEEN 1 AND 53", name="ck_raw_dhis2_epi_week"),
        db.CheckConstraint("value IS NULL OR value >= 0", name="ck_raw_dhis2_value_nonnegative"),
        db.CheckConstraint(
            "reporting_completeness IS NULL OR "
            "(reporting_completeness >= 0 AND reporting_completeness <= 100)",
            name="ck_raw_dhis2_reporting_completeness_percent",
        ),
        db.Index("ix_raw_dhis2_malaria_period", "period"),
        db.Index("ix_raw_dhis2_malaria_epi_year_week", "epi_year", "epi_week"),
        db.Index("ix_raw_dhis2_malaria_district_id", "district_id"),
        db.Index("ix_raw_dhis2_malaria_dhis2_orgunit_id", "dhis2_orgunit_id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    pipeline_run_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_runs.id"), nullable=False
    )
    dhis2_orgunit_id = db.Column(db.String(64), nullable=False)
    dhis2_orgunit_name = db.Column(db.String(160), nullable=True)
    district_id = db.Column(
        db.String(64), db.ForeignKey("district_reference.district_id"), nullable=True
    )
    data_element_id = db.Column(db.String(64), nullable=False)
    data_element_name = db.Column(db.String(200), nullable=True)
    period = db.Column(db.String(16), nullable=False)
    epi_year = db.Column(db.Integer, nullable=False)
    epi_week = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Numeric(14, 4), nullable=True)
    reporting_completeness = db.Column(db.Numeric(5, 2), nullable=True)
    raw_payload = db.Column(JSONB, nullable=True)
    extracted_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )

    pipeline_run = db.relationship("PipelineRun", back_populates="raw_dhis2_records")
    district = db.relationship("DistrictReference", back_populates="raw_dhis2_records")


class DistrictWeekFeature(db.Model):
    """Cleaned district-week analytical panel for modelling and summaries."""

    __tablename__ = "district_week_features"
    __table_args__ = (
        db.UniqueConstraint(
            "district_id", "epi_year", "epi_week", name="uq_dwf_district_epi_week"
        ),
        db.CheckConstraint("epi_week BETWEEN 1 AND 53", name="ck_dwf_epi_week"),
        db.CheckConstraint("confirmed_malaria_cases IS NULL OR confirmed_malaria_cases >= 0", name="ck_dwf_confirmed_cases_nonnegative"),
        db.CheckConstraint(
            "facility_reporting_completeness IS NULL OR "
            "(facility_reporting_completeness >= 0 AND facility_reporting_completeness <= 100)",
            name="ck_dwf_facility_reporting_completeness_percent",
        ),
        db.CheckConstraint(
            "time_series_completeness IS NULL OR "
            "(time_series_completeness >= 0 AND time_series_completeness <= 100)",
            name="ck_dwf_time_series_completeness_percent",
        ),
        db.CheckConstraint("population IS NULL OR population >= 0", name="ck_dwf_population_nonnegative"),
        db.CheckConstraint("incidence_per_1000 IS NULL OR incidence_per_1000 >= 0", name="ck_dwf_incidence_nonnegative"),
        db.CheckConstraint("month IS NULL OR month BETWEEN 1 AND 12", name="ck_dwf_month"),
        db.CheckConstraint("quarter IS NULL OR quarter BETWEEN 1 AND 4", name="ck_dwf_quarter"),
        db.CheckConstraint("descriptive_threshold IS NULL OR descriptive_threshold >= 0", name="ck_dwf_descriptive_threshold_nonnegative"),
        db.Index("ix_dwf_district_id", "district_id"),
        db.Index("ix_dwf_epi_year_week", "epi_year", "epi_week"),
        db.Index("ix_dwf_surge_observed", "surge_observed"),
    )

    id = db.Column(db.Integer, primary_key=True)
    district_id = db.Column(
        db.String(64), db.ForeignKey("district_reference.district_id"), nullable=False
    )
    epi_year = db.Column(db.Integer, nullable=False)
    epi_week = db.Column(db.Integer, nullable=False)
    week_start_date = db.Column(db.Date, nullable=True)
    week_end_date = db.Column(db.Date, nullable=True)
    confirmed_malaria_cases = db.Column(db.Integer, nullable=True)
    facility_reporting_completeness = db.Column(db.Numeric(5, 2), nullable=True)
    time_series_completeness = db.Column(db.Numeric(5, 2), nullable=True)
    population = db.Column(db.Integer, nullable=True)
    incidence_per_1000 = db.Column(db.Numeric(14, 6), nullable=True)
    rainfall_current = db.Column(db.Numeric(14, 6), nullable=True)
    rainfall_lag_4 = db.Column(db.Numeric(14, 6), nullable=True)
    rainfall_lag_8 = db.Column(db.Numeric(14, 6), nullable=True)
    rainfall_lag_12 = db.Column(db.Numeric(14, 6), nullable=True)
    temperature_current = db.Column(db.Numeric(8, 4), nullable=True)
    temperature_lag_4 = db.Column(db.Numeric(8, 4), nullable=True)
    temperature_lag_8 = db.Column(db.Numeric(8, 4), nullable=True)
    temperature_lag_12 = db.Column(db.Numeric(8, 4), nullable=True)
    humidity_current = db.Column(db.Numeric(8, 4), nullable=True)
    humidity_lag_4 = db.Column(db.Numeric(8, 4), nullable=True)
    humidity_lag_8 = db.Column(db.Numeric(8, 4), nullable=True)
    humidity_lag_12 = db.Column(db.Numeric(8, 4), nullable=True)
    cases_lag_1 = db.Column(db.Integer, nullable=True)
    cases_lag_2 = db.Column(db.Integer, nullable=True)
    cases_lag_4 = db.Column(db.Integer, nullable=True)
    cases_lag_8 = db.Column(db.Integer, nullable=True)
    cases_lag_12 = db.Column(db.Integer, nullable=True)
    month = db.Column(db.Integer, nullable=True)
    quarter = db.Column(db.Integer, nullable=True)
    season = db.Column(db.String(40), nullable=True)
    descriptive_threshold = db.Column(db.Numeric(14, 4), nullable=True)
    surge_observed = db.Column(db.Boolean, nullable=True)
    source_pipeline_run_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_runs.id"), nullable=True
    )
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=db.func.now(),
        onupdate=db.func.now(),
    )

    district = db.relationship("DistrictReference", back_populates="week_features")
    source_pipeline_run = db.relationship(
        "PipelineRun", back_populates="district_week_features"
    )


class DistrictWeekClimate(db.Model):
    """Processed current-week climate summaries keyed by district_id and ISO week."""

    __tablename__ = "district_week_climate"
    __table_args__ = (
        db.UniqueConstraint(
            "district_id", "epi_year", "epi_week", name="uq_dwc_district_epi_week"
        ),
        db.CheckConstraint("epi_week BETWEEN 1 AND 53", name="ck_dwc_epi_week"),
        db.CheckConstraint(
            "rainfall_current IS NULL OR rainfall_current >= 0",
            name="ck_dwc_rainfall_current_nonnegative",
        ),
        db.Index("ix_dwc_district_id", "district_id"),
        db.Index("ix_dwc_epi_year_week", "epi_year", "epi_week"),
        db.Index("ix_dwc_pipeline_run_id", "pipeline_run_id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    district_id = db.Column(
        db.String(64), db.ForeignKey("district_reference.district_id"), nullable=False
    )
    epi_year = db.Column(db.Integer, nullable=False)
    epi_week = db.Column(db.Integer, nullable=False)
    week_start_date = db.Column(db.Date, nullable=False)
    week_end_date = db.Column(db.Date, nullable=False)
    rainfall_current = db.Column(db.Numeric(14, 6), nullable=True)
    temperature_current = db.Column(db.Numeric(8, 4), nullable=True)
    humidity_current = db.Column(db.Numeric(8, 4), nullable=True)
    source_rainfall = db.Column(db.String(160), nullable=True)
    source_temperature = db.Column(db.String(160), nullable=True)
    source_humidity = db.Column(db.String(160), nullable=True)
    aggregation_method = db.Column(db.String(120), nullable=False)
    pipeline_run_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_runs.id"), nullable=False
    )
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=db.func.now(),
        onupdate=db.func.now(),
    )

    district = db.relationship("DistrictReference", backref="climate_weeks")
    pipeline_run = db.relationship("PipelineRun", back_populates="district_week_climate")


class ForecastOutput(db.Model):
    """Model-generated forecasts and surge classifications."""

    __tablename__ = "forecast_outputs"
    __table_args__ = (
        db.CheckConstraint(
            _in_constraint("model_name", APPROVED_MODEL_NAMES),
            name="ck_forecast_outputs_model_name",
        ),
        db.CheckConstraint(
            _in_constraint("model_scope", MODEL_SCOPES),
            name="ck_forecast_outputs_model_scope",
        ),
        db.CheckConstraint(
            _in_constraint("horizon_weeks", FORECAST_HORIZONS),
            name="ck_forecast_outputs_horizon_weeks",
        ),
        db.CheckConstraint(
            _in_constraint("risk_level", RISK_LEVELS),
            name="ck_forecast_outputs_risk_level",
        ),
        db.CheckConstraint(
            "forecast_origin_epi_week BETWEEN 1 AND 53",
            name="ck_forecast_outputs_origin_epi_week",
        ),
        db.CheckConstraint(
            "target_epi_week BETWEEN 1 AND 53",
            name="ck_forecast_outputs_target_epi_week",
        ),
        db.CheckConstraint("predicted_cases IS NULL OR predicted_cases >= 0", name="ck_forecast_outputs_predicted_cases_nonnegative"),
        db.CheckConstraint(
            "prediction_interval_lower IS NULL OR prediction_interval_lower >= 0",
            name="ck_forecast_outputs_pi_lower_nonnegative",
        ),
        db.CheckConstraint(
            "prediction_interval_upper IS NULL OR prediction_interval_upper >= 0",
            name="ck_forecast_outputs_pi_upper_nonnegative",
        ),
        db.CheckConstraint(
            "prediction_interval_lower IS NULL OR prediction_interval_upper IS NULL OR "
            "prediction_interval_lower <= prediction_interval_upper",
            name="ck_forecast_outputs_prediction_interval_order",
        ),
        db.CheckConstraint(
            "surge_probability IS NULL OR "
            "(surge_probability >= 0 AND surge_probability <= 1)",
            name="ck_forecast_outputs_surge_probability",
        ),
        db.CheckConstraint("threshold_used IS NULL OR threshold_used >= 0", name="ck_forecast_outputs_threshold_used_nonnegative"),
        db.Index("ix_forecast_outputs_district_id", "district_id"),
        db.Index("ix_forecast_outputs_model_name", "model_name"),
        db.Index("ix_forecast_outputs_horizon_weeks", "horizon_weeks"),
        db.Index(
            "ix_forecast_outputs_target_epi",
            "target_epi_year",
            "target_epi_week",
        ),
        db.Index(
            "ix_forecast_outputs_district_target_epi",
            "district_id",
            "target_epi_year",
            "target_epi_week",
        ),
        db.Index("ix_forecast_outputs_risk_level", "risk_level"),
    )

    id = db.Column(db.Integer, primary_key=True)
    pipeline_run_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_runs.id"), nullable=False
    )
    district_id = db.Column(
        db.String(64), db.ForeignKey("district_reference.district_id"), nullable=False
    )
    model_name = db.Column(db.String(40), nullable=False)
    model_scope = db.Column(db.String(40), nullable=False)
    forecast_origin_epi_year = db.Column(db.Integer, nullable=False)
    forecast_origin_epi_week = db.Column(db.Integer, nullable=False)
    forecast_origin_date = db.Column(db.Date, nullable=True)
    horizon_weeks = db.Column(db.Integer, nullable=False)
    target_epi_year = db.Column(db.Integer, nullable=False)
    target_epi_week = db.Column(db.Integer, nullable=False)
    target_week_start_date = db.Column(db.Date, nullable=True)
    predicted_cases = db.Column(db.Numeric(14, 4), nullable=True)
    prediction_interval_lower = db.Column(db.Numeric(14, 4), nullable=True)
    prediction_interval_upper = db.Column(db.Numeric(14, 4), nullable=True)
    surge_probability = db.Column(db.Numeric(6, 5), nullable=True)
    predicted_surge_flag = db.Column(db.Boolean, nullable=True)
    risk_level = db.Column(db.String(20), nullable=False)
    threshold_used = db.Column(db.Numeric(14, 4), nullable=True)
    threshold_method = db.Column(db.String(120), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )

    pipeline_run = db.relationship("PipelineRun", back_populates="forecast_outputs")
    district = db.relationship("DistrictReference", back_populates="forecast_outputs")


class ModelPerformance(db.Model):
    """Rolling-origin / forward-chaining model evaluation results."""

    __tablename__ = "model_performance"
    __table_args__ = (
        db.CheckConstraint(
            _in_constraint("model_name", APPROVED_MODEL_NAMES),
            name="ck_model_performance_model_name",
        ),
        db.CheckConstraint(
            _in_constraint("horizon_weeks", FORECAST_HORIZONS),
            name="ck_model_performance_horizon_weeks",
        ),
        db.CheckConstraint(
            _in_constraint("model_scope", MODEL_SCOPES),
            name="ck_model_performance_model_scope",
        ),
        db.CheckConstraint(
            _in_constraint("validation_method", VALIDATION_METHODS),
            name="ck_model_performance_validation_method",
        ),
        db.CheckConstraint(
            "evaluation_start_epi_week BETWEEN 1 AND 53",
            name="ck_model_performance_start_epi_week",
        ),
        db.CheckConstraint(
            "evaluation_end_epi_week BETWEEN 1 AND 53",
            name="ck_model_performance_end_epi_week",
        ),
        db.CheckConstraint("mae IS NULL OR mae >= 0", name="ck_model_performance_mae_nonnegative"),
        db.CheckConstraint("rmse IS NULL OR rmse >= 0", name="ck_model_performance_rmse_nonnegative"),
        db.CheckConstraint("mape IS NULL OR mape >= 0", name="ck_model_performance_mape_nonnegative"),
        db.CheckConstraint("wape IS NULL OR wape >= 0", name="ck_model_performance_wape_nonnegative"),
        db.CheckConstraint(
            "sensitivity IS NULL OR (sensitivity >= 0 AND sensitivity <= 1)",
            name="ck_model_performance_sensitivity",
        ),
        db.CheckConstraint(
            "specificity IS NULL OR (specificity >= 0 AND specificity <= 1)",
            name="ck_model_performance_specificity",
        ),
        db.CheckConstraint(
            "precision IS NULL OR (precision >= 0 AND precision <= 1)",
            name="ck_model_performance_precision",
        ),
        db.CheckConstraint(
            "f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)",
            name="ck_model_performance_f1",
        ),
        db.CheckConstraint(
            "auc IS NULL OR (auc >= 0 AND auc <= 1)",
            name="ck_model_performance_auc",
        ),
        db.Index("ix_model_performance_model_name", "model_name"),
        db.Index("ix_model_performance_model_scope", "model_scope"),
        db.Index("ix_model_performance_horizon_weeks", "horizon_weeks"),
        db.Index("ix_model_performance_district_id", "district_id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    pipeline_run_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_runs.id"), nullable=False
    )
    model_name = db.Column(db.String(40), nullable=False)
    model_scope = db.Column(db.String(40), nullable=False)
    horizon_weeks = db.Column(db.Integer, nullable=False)
    evaluation_start_epi_year = db.Column(db.Integer, nullable=False)
    evaluation_start_epi_week = db.Column(db.Integer, nullable=False)
    evaluation_end_epi_year = db.Column(db.Integer, nullable=False)
    evaluation_end_epi_week = db.Column(db.Integer, nullable=False)
    district_id = db.Column(
        db.String(64), db.ForeignKey("district_reference.district_id"), nullable=True
    )
    mae = db.Column(db.Numeric(14, 6), nullable=True)
    rmse = db.Column(db.Numeric(14, 6), nullable=True)
    mape = db.Column(db.Numeric(14, 6), nullable=True)
    wape = db.Column(db.Numeric(14, 6), nullable=True)
    r_squared = db.Column(db.Numeric(14, 6), nullable=True)
    sensitivity = db.Column(db.Numeric(6, 5), nullable=True)
    specificity = db.Column(db.Numeric(6, 5), nullable=True)
    precision = db.Column(db.Numeric(6, 5), nullable=True)
    f1_score = db.Column(db.Numeric(6, 5), nullable=True)
    auc = db.Column(db.Numeric(6, 5), nullable=True)
    validation_method = db.Column(
        db.String(30),
        nullable=False,
        default="rolling_origin",
        server_default="rolling_origin",
    )
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )

    pipeline_run = db.relationship("PipelineRun", back_populates="model_performance")
    district = db.relationship("DistrictReference", back_populates="model_performance")
