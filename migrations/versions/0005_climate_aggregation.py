"""add district week climate aggregation support

Revision ID: 0005_climate_aggregation
Revises: 0004_ubos_population
Create Date: 2026-07-06

"""
from alembic import op
import sqlalchemy as sa


revision = "0005_climate_aggregation"
down_revision = "0004_ubos_population"
branch_labels = None
depends_on = None


PIPELINE_RUN_TYPES = (
    "dhis2_extraction",
    "preprocessing",
    "climate_ingestion",
    "climate_aggregation",
    "feature_engineering",
    "model_training",
    "forecast_generation",
    "report_export",
)

PREVIOUS_PIPELINE_RUN_TYPES = tuple(
    value for value in PIPELINE_RUN_TYPES if value != "climate_aggregation"
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

    op.create_table(
        "district_week_climate",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("district_id", sa.String(length=64), nullable=False),
        sa.Column("epi_year", sa.Integer(), nullable=False),
        sa.Column("epi_week", sa.Integer(), nullable=False),
        sa.Column("week_start_date", sa.Date(), nullable=False),
        sa.Column("week_end_date", sa.Date(), nullable=False),
        sa.Column("rainfall_current", sa.Numeric(14, 6), nullable=True),
        sa.Column("temperature_current", sa.Numeric(8, 4), nullable=True),
        sa.Column("humidity_current", sa.Numeric(8, 4), nullable=True),
        sa.Column("source_rainfall", sa.String(length=160), nullable=True),
        sa.Column("source_temperature", sa.String(length=160), nullable=True),
        sa.Column("source_humidity", sa.String(length=160), nullable=True),
        sa.Column("aggregation_method", sa.String(length=120), nullable=False),
        sa.Column("pipeline_run_id", sa.Integer(), nullable=False),
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
        sa.CheckConstraint("epi_week BETWEEN 1 AND 53", name="ck_dwc_epi_week"),
        sa.CheckConstraint(
            "rainfall_current IS NULL OR rainfall_current >= 0",
            name="ck_dwc_rainfall_current_nonnegative",
        ),
        sa.ForeignKeyConstraint(["district_id"], ["district_reference.district_id"]),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "district_id", "epi_year", "epi_week", name="uq_dwc_district_epi_week"
        ),
    )
    op.create_index("ix_dwc_district_id", "district_week_climate", ["district_id"])
    op.create_index(
        "ix_dwc_epi_year_week", "district_week_climate", ["epi_year", "epi_week"]
    )
    op.create_index(
        "ix_dwc_pipeline_run_id", "district_week_climate", ["pipeline_run_id"]
    )


def downgrade():
    op.drop_index("ix_dwc_pipeline_run_id", table_name="district_week_climate")
    op.drop_index("ix_dwc_epi_year_week", table_name="district_week_climate")
    op.drop_index("ix_dwc_district_id", table_name="district_week_climate")
    op.drop_table("district_week_climate")

    op.drop_constraint("ck_pipeline_runs_run_type", "pipeline_runs", type_="check")
    op.create_check_constraint(
        "ck_pipeline_runs_run_type",
        "pipeline_runs",
        _in_constraint("run_type", PREVIOUS_PIPELINE_RUN_TYPES),
    )
