"""add DHIS2 reporting completeness pipeline type

Revision ID: 0006_reporting_completeness_type
Revises: 0005_climate_aggregation
Create Date: 2026-07-07

"""
from alembic import op


revision = "0006_reporting_completeness_type"
down_revision = "0005_climate_aggregation"
branch_labels = None
depends_on = None


PIPELINE_RUN_TYPES = (
    "dhis2_extraction",
    "dhis2_reporting_completeness",
    "preprocessing",
    "climate_ingestion",
    "climate_aggregation",
    "feature_engineering",
    "model_training",
    "forecast_generation",
    "report_export",
)

PREVIOUS_PIPELINE_RUN_TYPES = tuple(
    value for value in PIPELINE_RUN_TYPES if value != "dhis2_reporting_completeness"
)


def _in_constraint(column_name, values):
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column_name} IN ({rendered})"


def upgrade():
    op.drop_constraint("ck_pipeline_runs_run_type", "pipeline_runs", type_="check")
    op.create_check_constraint(
        "ck_pipeline_runs_run_type",
        "pipeline_runs",
        _in_constraint("run_type", PIPELINE_RUN_TYPES),
    )


def downgrade():
    op.drop_constraint("ck_pipeline_runs_run_type", "pipeline_runs", type_="check")
    op.create_check_constraint(
        "ck_pipeline_runs_run_type",
        "pipeline_runs",
        _in_constraint("run_type", PREVIOUS_PIPELINE_RUN_TYPES),
    )
