"""add ubos population denominators

Revision ID: 0004_ubos_population
Revises: 0003_add_cases_lag_2
Create Date: 2026-07-06

"""
from alembic import op
import sqlalchemy as sa


revision = "0004_ubos_population"
down_revision = "0003_add_cases_lag_2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ubos_population",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("district_id", sa.String(length=64), nullable=False),
        sa.Column("district_name", sa.String(length=160), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("population", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=160), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("population > 0", name="ck_ubos_population_positive"),
        sa.CheckConstraint("year >= 1900 AND year <= 2100", name="ck_ubos_population_year"),
        sa.ForeignKeyConstraint(["district_id"], ["district_reference.district_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("district_id", "year", name="uq_ubos_population_district_year"),
    )
    op.create_index("ix_ubos_population_district_id", "ubos_population", ["district_id"])
    op.create_index("ix_ubos_population_year", "ubos_population", ["year"])


def downgrade():
    op.drop_index("ix_ubos_population_year", table_name="ubos_population")
    op.drop_index("ix_ubos_population_district_id", table_name="ubos_population")
    op.drop_table("ubos_population")
