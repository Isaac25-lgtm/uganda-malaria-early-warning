"""add cases_lag_2 to district_week_features

Revision ID: 0003_add_cases_lag_2
Revises: 0002_core_research_tables
Create Date: 2026-07-06

"""
from alembic import op
import sqlalchemy as sa


revision = "0003_add_cases_lag_2"
down_revision = "0002_core_research_tables"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "district_week_features",
        sa.Column("cases_lag_2", sa.Integer(), nullable=True),
    )


def downgrade():
    op.drop_column("district_week_features", "cases_lag_2")
