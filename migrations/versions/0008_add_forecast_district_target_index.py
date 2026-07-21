"""add forecast district target composite index

Revision ID: 0008_forecast_index
Revises: 0007_climate_lag_creation_type
Create Date: 2026-07-07

"""
from alembic import op


revision = "0008_forecast_index"
down_revision = "0007_climate_lag_creation_type"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "ix_forecast_outputs_district_target_epi",
        "forecast_outputs",
        ["district_id", "target_epi_year", "target_epi_week"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_forecast_outputs_district_target_epi",
        table_name="forecast_outputs",
    )
