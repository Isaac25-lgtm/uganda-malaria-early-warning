"""create core research data contract tables

Revision ID: 0002_core_research_tables
Revises: 0001_initial_users
Create Date: 2026-07-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0002_core_research_tables"
down_revision = "0001_initial_users"
branch_labels = None
depends_on = None


APPROVED_MODEL_NAMES = ("Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost")
FORECAST_HORIZONS = (4, 8, 12)
RISK_LEVELS = ("Low", "Moderate", "High", "Very High", "No data")
VALIDATION_METHODS = ("rolling_origin", "forward_chaining")
MODEL_SCOPES = ("national_pooled", "regional_pooled", "district_specific", "other")
PIPELINE_RUN_TYPES = (
    "dhis2_extraction",
    "preprocessing",
    "climate_ingestion",
    "feature_engineering",
    "model_training",
    "forecast_generation",
    "report_export",
)
PIPELINE_STATUSES = ("pending", "running", "success", "failed")


def _in_constraint(column_name, values):
    rendered = ", ".join(
        f"'{value}'" if isinstance(value, str) else str(value) for value in values
    )
    return f"{column_name} IN ({rendered})"


def upgrade():
    op.create_table(
        "district_reference",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("district_id", sa.String(length=64), nullable=False),
        sa.Column("district_name", sa.String(length=160), nullable=False),
        sa.Column("dhis2_orgunit_id", sa.String(length=64), nullable=True),
        sa.Column("dhis2_orgunit_name", sa.String(length=160), nullable=True),
        sa.Column("ubos_code", sa.String(length=64), nullable=True),
        sa.Column("region", sa.String(length=120), nullable=True),
        sa.Column("subregion", sa.String(length=120), nullable=True),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("district_id"),
    )
    op.create_index(
        "ix_district_reference_district_id",
        "district_reference",
        ["district_id"],
        unique=False,
    )
    op.create_index(
        "ix_district_reference_dhis2_orgunit_id",
        "district_reference",
        ["dhis2_orgunit_id"],
        unique=False,
    )
    op.create_index(
        "ix_district_reference_ubos_code",
        "district_reference",
        ["ubos_code"],
        unique=False,
    )

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_type", sa.String(length=40), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_name", sa.String(length=160), nullable=True),
        sa.Column("source_version_or_url", sa.String(length=500), nullable=True),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("row_count_raw", sa.Integer(), nullable=True),
        sa.Column("row_count_processed", sa.Integer(), nullable=True),
        sa.Column("warning_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            _in_constraint("run_type", PIPELINE_RUN_TYPES),
            name="ck_pipeline_runs_run_type",
        ),
        sa.CheckConstraint(
            _in_constraint("status", PIPELINE_STATUSES),
            name="ck_pipeline_runs_status",
        ),
        sa.CheckConstraint(
            "row_count_raw IS NULL OR row_count_raw >= 0",
            name="ck_pipeline_runs_row_count_raw_nonnegative",
        ),
        sa.CheckConstraint(
            "row_count_processed IS NULL OR row_count_processed >= 0",
            name="ck_pipeline_runs_row_count_processed_nonnegative",
        ),
        sa.CheckConstraint(
            "warning_count IS NULL OR warning_count >= 0",
            name="ck_pipeline_runs_warning_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_runs_run_type", "pipeline_runs", ["run_type"])
    op.create_index("ix_pipeline_runs_status", "pipeline_runs", ["status"])
    op.create_index("ix_pipeline_runs_started_at", "pipeline_runs", ["started_at"])

    op.create_table(
        "raw_dhis2_malaria",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pipeline_run_id", sa.Integer(), nullable=False),
        sa.Column("dhis2_orgunit_id", sa.String(length=64), nullable=False),
        sa.Column("dhis2_orgunit_name", sa.String(length=160), nullable=True),
        sa.Column("district_id", sa.String(length=64), nullable=True),
        sa.Column("data_element_id", sa.String(length=64), nullable=False),
        sa.Column("data_element_name", sa.String(length=200), nullable=True),
        sa.Column("period", sa.String(length=16), nullable=False),
        sa.Column("epi_year", sa.Integer(), nullable=False),
        sa.Column("epi_week", sa.Integer(), nullable=False),
        sa.Column("value", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("reporting_completeness", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("epi_week BETWEEN 1 AND 53", name="ck_raw_dhis2_epi_week"),
        sa.CheckConstraint(
            "value IS NULL OR value >= 0",
            name="ck_raw_dhis2_value_nonnegative",
        ),
        sa.CheckConstraint(
            "reporting_completeness IS NULL OR "
            "(reporting_completeness >= 0 AND reporting_completeness <= 100)",
            name="ck_raw_dhis2_reporting_completeness_percent",
        ),
        sa.ForeignKeyConstraint(["district_id"], ["district_reference.district_id"]),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "dhis2_orgunit_id",
            "data_element_id",
            "period",
            name="uq_raw_dhis2_run_orgunit_dataelement_period",
        ),
    )
    op.create_index("ix_raw_dhis2_malaria_period", "raw_dhis2_malaria", ["period"])
    op.create_index(
        "ix_raw_dhis2_malaria_epi_year_week",
        "raw_dhis2_malaria",
        ["epi_year", "epi_week"],
    )
    op.create_index(
        "ix_raw_dhis2_malaria_district_id", "raw_dhis2_malaria", ["district_id"]
    )
    op.create_index(
        "ix_raw_dhis2_malaria_dhis2_orgunit_id",
        "raw_dhis2_malaria",
        ["dhis2_orgunit_id"],
    )

    op.create_table(
        "district_week_features",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("district_id", sa.String(length=64), nullable=False),
        sa.Column("epi_year", sa.Integer(), nullable=False),
        sa.Column("epi_week", sa.Integer(), nullable=False),
        sa.Column("week_start_date", sa.Date(), nullable=True),
        sa.Column("week_end_date", sa.Date(), nullable=True),
        sa.Column("confirmed_malaria_cases", sa.Integer(), nullable=True),
        sa.Column(
            "facility_reporting_completeness",
            sa.Numeric(precision=5, scale=2),
            nullable=True,
        ),
        sa.Column(
            "time_series_completeness",
            sa.Numeric(precision=5, scale=2),
            nullable=True,
        ),
        sa.Column("population", sa.Integer(), nullable=True),
        sa.Column("incidence_per_1000", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("rainfall_current", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("rainfall_lag_4", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("rainfall_lag_8", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("rainfall_lag_12", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("temperature_current", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("temperature_lag_4", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("temperature_lag_8", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("temperature_lag_12", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("humidity_current", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("humidity_lag_4", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("humidity_lag_8", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("humidity_lag_12", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("cases_lag_1", sa.Integer(), nullable=True),
        sa.Column("cases_lag_4", sa.Integer(), nullable=True),
        sa.Column("cases_lag_8", sa.Integer(), nullable=True),
        sa.Column("cases_lag_12", sa.Integer(), nullable=True),
        sa.Column("month", sa.Integer(), nullable=True),
        sa.Column("quarter", sa.Integer(), nullable=True),
        sa.Column("season", sa.String(length=40), nullable=True),
        sa.Column("descriptive_threshold", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("surge_observed", sa.Boolean(), nullable=True),
        sa.Column("source_pipeline_run_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("epi_week BETWEEN 1 AND 53", name="ck_dwf_epi_week"),
        sa.CheckConstraint(
            "confirmed_malaria_cases IS NULL OR confirmed_malaria_cases >= 0",
            name="ck_dwf_confirmed_cases_nonnegative",
        ),
        sa.CheckConstraint(
            "facility_reporting_completeness IS NULL OR "
            "(facility_reporting_completeness >= 0 AND facility_reporting_completeness <= 100)",
            name="ck_dwf_facility_reporting_completeness_percent",
        ),
        sa.CheckConstraint(
            "time_series_completeness IS NULL OR "
            "(time_series_completeness >= 0 AND time_series_completeness <= 100)",
            name="ck_dwf_time_series_completeness_percent",
        ),
        sa.CheckConstraint(
            "population IS NULL OR population >= 0",
            name="ck_dwf_population_nonnegative",
        ),
        sa.CheckConstraint(
            "incidence_per_1000 IS NULL OR incidence_per_1000 >= 0",
            name="ck_dwf_incidence_nonnegative",
        ),
        sa.CheckConstraint("month IS NULL OR month BETWEEN 1 AND 12", name="ck_dwf_month"),
        sa.CheckConstraint(
            "quarter IS NULL OR quarter BETWEEN 1 AND 4",
            name="ck_dwf_quarter",
        ),
        sa.CheckConstraint(
            "descriptive_threshold IS NULL OR descriptive_threshold >= 0",
            name="ck_dwf_descriptive_threshold_nonnegative",
        ),
        sa.ForeignKeyConstraint(["district_id"], ["district_reference.district_id"]),
        sa.ForeignKeyConstraint(["source_pipeline_run_id"], ["pipeline_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("district_id", "epi_year", "epi_week", name="uq_dwf_district_epi_week"),
    )
    op.create_index("ix_dwf_district_id", "district_week_features", ["district_id"])
    op.create_index(
        "ix_dwf_epi_year_week",
        "district_week_features",
        ["epi_year", "epi_week"],
    )
    op.create_index("ix_dwf_surge_observed", "district_week_features", ["surge_observed"])

    op.create_table(
        "forecast_outputs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pipeline_run_id", sa.Integer(), nullable=False),
        sa.Column("district_id", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=40), nullable=False),
        sa.Column("model_scope", sa.String(length=40), nullable=False),
        sa.Column("forecast_origin_epi_year", sa.Integer(), nullable=False),
        sa.Column("forecast_origin_epi_week", sa.Integer(), nullable=False),
        sa.Column("forecast_origin_date", sa.Date(), nullable=True),
        sa.Column("horizon_weeks", sa.Integer(), nullable=False),
        sa.Column("target_epi_year", sa.Integer(), nullable=False),
        sa.Column("target_epi_week", sa.Integer(), nullable=False),
        sa.Column("target_week_start_date", sa.Date(), nullable=True),
        sa.Column("predicted_cases", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column(
            "prediction_interval_lower",
            sa.Numeric(precision=14, scale=4),
            nullable=True,
        ),
        sa.Column(
            "prediction_interval_upper",
            sa.Numeric(precision=14, scale=4),
            nullable=True,
        ),
        sa.Column("surge_probability", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("predicted_surge_flag", sa.Boolean(), nullable=True),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("threshold_used", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("threshold_method", sa.String(length=120), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            _in_constraint("model_name", APPROVED_MODEL_NAMES),
            name="ck_forecast_outputs_model_name",
        ),
        sa.CheckConstraint(
            _in_constraint("model_scope", MODEL_SCOPES),
            name="ck_forecast_outputs_model_scope",
        ),
        sa.CheckConstraint(
            _in_constraint("horizon_weeks", FORECAST_HORIZONS),
            name="ck_forecast_outputs_horizon_weeks",
        ),
        sa.CheckConstraint(
            _in_constraint("risk_level", RISK_LEVELS),
            name="ck_forecast_outputs_risk_level",
        ),
        sa.CheckConstraint(
            "forecast_origin_epi_week BETWEEN 1 AND 53",
            name="ck_forecast_outputs_origin_epi_week",
        ),
        sa.CheckConstraint(
            "target_epi_week BETWEEN 1 AND 53",
            name="ck_forecast_outputs_target_epi_week",
        ),
        sa.CheckConstraint(
            "predicted_cases IS NULL OR predicted_cases >= 0",
            name="ck_forecast_outputs_predicted_cases_nonnegative",
        ),
        sa.CheckConstraint(
            "prediction_interval_lower IS NULL OR prediction_interval_lower >= 0",
            name="ck_forecast_outputs_pi_lower_nonnegative",
        ),
        sa.CheckConstraint(
            "prediction_interval_upper IS NULL OR prediction_interval_upper >= 0",
            name="ck_forecast_outputs_pi_upper_nonnegative",
        ),
        sa.CheckConstraint(
            "prediction_interval_lower IS NULL OR prediction_interval_upper IS NULL OR "
            "prediction_interval_lower <= prediction_interval_upper",
            name="ck_forecast_outputs_prediction_interval_order",
        ),
        sa.CheckConstraint(
            "surge_probability IS NULL OR "
            "(surge_probability >= 0 AND surge_probability <= 1)",
            name="ck_forecast_outputs_surge_probability",
        ),
        sa.CheckConstraint(
            "threshold_used IS NULL OR threshold_used >= 0",
            name="ck_forecast_outputs_threshold_used_nonnegative",
        ),
        sa.ForeignKeyConstraint(["district_id"], ["district_reference.district_id"]),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecast_outputs_district_id", "forecast_outputs", ["district_id"])
    op.create_index("ix_forecast_outputs_model_name", "forecast_outputs", ["model_name"])
    op.create_index("ix_forecast_outputs_horizon_weeks", "forecast_outputs", ["horizon_weeks"])
    op.create_index(
        "ix_forecast_outputs_target_epi",
        "forecast_outputs",
        ["target_epi_year", "target_epi_week"],
    )
    op.create_index("ix_forecast_outputs_risk_level", "forecast_outputs", ["risk_level"])

    op.create_table(
        "model_performance",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pipeline_run_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=40), nullable=False),
        sa.Column("model_scope", sa.String(length=40), nullable=False),
        sa.Column("horizon_weeks", sa.Integer(), nullable=False),
        sa.Column("evaluation_start_epi_year", sa.Integer(), nullable=False),
        sa.Column("evaluation_start_epi_week", sa.Integer(), nullable=False),
        sa.Column("evaluation_end_epi_year", sa.Integer(), nullable=False),
        sa.Column("evaluation_end_epi_week", sa.Integer(), nullable=False),
        sa.Column("district_id", sa.String(length=64), nullable=True),
        sa.Column("mae", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("rmse", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("mape", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("wape", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("r_squared", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("sensitivity", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("specificity", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("precision", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("f1_score", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("auc", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column(
            "validation_method",
            sa.String(length=30),
            server_default="rolling_origin",
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            _in_constraint("model_name", APPROVED_MODEL_NAMES),
            name="ck_model_performance_model_name",
        ),
        sa.CheckConstraint(
            _in_constraint("model_scope", MODEL_SCOPES),
            name="ck_model_performance_model_scope",
        ),
        sa.CheckConstraint(
            _in_constraint("horizon_weeks", FORECAST_HORIZONS),
            name="ck_model_performance_horizon_weeks",
        ),
        sa.CheckConstraint(
            _in_constraint("validation_method", VALIDATION_METHODS),
            name="ck_model_performance_validation_method",
        ),
        sa.CheckConstraint(
            "evaluation_start_epi_week BETWEEN 1 AND 53",
            name="ck_model_performance_start_epi_week",
        ),
        sa.CheckConstraint(
            "evaluation_end_epi_week BETWEEN 1 AND 53",
            name="ck_model_performance_end_epi_week",
        ),
        sa.CheckConstraint("mae IS NULL OR mae >= 0", name="ck_model_performance_mae_nonnegative"),
        sa.CheckConstraint("rmse IS NULL OR rmse >= 0", name="ck_model_performance_rmse_nonnegative"),
        sa.CheckConstraint("mape IS NULL OR mape >= 0", name="ck_model_performance_mape_nonnegative"),
        sa.CheckConstraint("wape IS NULL OR wape >= 0", name="ck_model_performance_wape_nonnegative"),
        sa.CheckConstraint(
            "sensitivity IS NULL OR (sensitivity >= 0 AND sensitivity <= 1)",
            name="ck_model_performance_sensitivity",
        ),
        sa.CheckConstraint(
            "specificity IS NULL OR (specificity >= 0 AND specificity <= 1)",
            name="ck_model_performance_specificity",
        ),
        sa.CheckConstraint(
            "precision IS NULL OR (precision >= 0 AND precision <= 1)",
            name="ck_model_performance_precision",
        ),
        sa.CheckConstraint(
            "f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)",
            name="ck_model_performance_f1",
        ),
        sa.CheckConstraint(
            "auc IS NULL OR (auc >= 0 AND auc <= 1)",
            name="ck_model_performance_auc",
        ),
        sa.ForeignKeyConstraint(["district_id"], ["district_reference.district_id"]),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_performance_model_name", "model_performance", ["model_name"])
    op.create_index("ix_model_performance_model_scope", "model_performance", ["model_scope"])
    op.create_index(
        "ix_model_performance_horizon_weeks",
        "model_performance",
        ["horizon_weeks"],
    )
    op.create_index("ix_model_performance_district_id", "model_performance", ["district_id"])


def downgrade():
    op.drop_index("ix_model_performance_district_id", table_name="model_performance")
    op.drop_index("ix_model_performance_horizon_weeks", table_name="model_performance")
    op.drop_index("ix_model_performance_model_scope", table_name="model_performance")
    op.drop_index("ix_model_performance_model_name", table_name="model_performance")
    op.drop_table("model_performance")

    op.drop_index("ix_forecast_outputs_risk_level", table_name="forecast_outputs")
    op.drop_index("ix_forecast_outputs_target_epi", table_name="forecast_outputs")
    op.drop_index("ix_forecast_outputs_horizon_weeks", table_name="forecast_outputs")
    op.drop_index("ix_forecast_outputs_model_name", table_name="forecast_outputs")
    op.drop_index("ix_forecast_outputs_district_id", table_name="forecast_outputs")
    op.drop_table("forecast_outputs")

    op.drop_index("ix_dwf_surge_observed", table_name="district_week_features")
    op.drop_index("ix_dwf_epi_year_week", table_name="district_week_features")
    op.drop_index("ix_dwf_district_id", table_name="district_week_features")
    op.drop_table("district_week_features")

    op.drop_index("ix_raw_dhis2_malaria_dhis2_orgunit_id", table_name="raw_dhis2_malaria")
    op.drop_index("ix_raw_dhis2_malaria_district_id", table_name="raw_dhis2_malaria")
    op.drop_index("ix_raw_dhis2_malaria_epi_year_week", table_name="raw_dhis2_malaria")
    op.drop_index("ix_raw_dhis2_malaria_period", table_name="raw_dhis2_malaria")
    op.drop_table("raw_dhis2_malaria")

    op.drop_index("ix_pipeline_runs_started_at", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_status", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_run_type", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")

    op.drop_index("ix_district_reference_ubos_code", table_name="district_reference")
    op.drop_index("ix_district_reference_dhis2_orgunit_id", table_name="district_reference")
    op.drop_index("ix_district_reference_district_id", table_name="district_reference")
    op.drop_table("district_reference")
