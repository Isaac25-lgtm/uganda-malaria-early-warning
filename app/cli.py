"""Custom Flask CLI commands."""
from __future__ import annotations

import click
from flask import Flask
from flask.cli import with_appcontext

from app.extensions import db
from app.services.project_state import PROJECT_STATE_SUMMARY_PATH, build_project_state_summary

CORE_RESEARCH_TABLES = (
    "district_reference",
    "pipeline_runs",
    "raw_dhis2_malaria",
    "district_week_features",
    "forecast_outputs",
    "model_performance",
)
SUPPLEMENTARY_REFERENCE_TABLES = ("ubos_population",)
PROCESSED_SUPPORT_TABLES = ("district_week_climate",)


def register_cli(app: Flask) -> None:
    from app.commands.climate import climate_cli
    from app.commands.dashboard import dashboard_cli
    from app.commands.dhis2 import dhis2_cli
    from app.commands.forecasting import forecasting_cli
    from app.commands.modelling import modelling_cli
    from app.commands.population import population_cli
    from app.commands.preprocessing import preprocessing_cli
    from app.commands.reference import reference_cli

    app.cli.add_command(verify_data_contract)
    app.cli.add_command(verify_project_state)
    app.cli.add_command(reference_cli)
    app.cli.add_command(dhis2_cli)
    app.cli.add_command(preprocessing_cli)
    app.cli.add_command(population_cli)
    app.cli.add_command(climate_cli)
    app.cli.add_command(modelling_cli)
    app.cli.add_command(forecasting_cli)
    app.cli.add_command(dashboard_cli)

@click.command("verify-data-contract")
@with_appcontext
def verify_data_contract() -> None:
    """Verify the core research table contract is registered in SQLAlchemy."""
    # Importing app.models registers all mapped classes with db.metadata.
    from app import models  # noqa: F401

    metadata_tables = db.metadata.tables
    required_tables = (
        CORE_RESEARCH_TABLES + SUPPLEMENTARY_REFERENCE_TABLES + PROCESSED_SUPPORT_TABLES
    )
    missing = [name for name in required_tables if name not in metadata_tables]
    if missing:
        raise click.ClickException(
            "Missing core research table metadata: " + ", ".join(missing)
        )

    wind_fields: list[str] = []
    for table_name in required_tables:
        table = metadata_tables[table_name]
        for column in table.columns:
            if "wind" in column.name.lower():
                wind_fields.append(f"{table_name}.{column.name}")

    if wind_fields:
        raise click.ClickException(
            "Unsupported wind-speed fields found: " + ", ".join(wind_fields)
        )

    for table_name in required_tables:
        table = metadata_tables[table_name]
        click.echo(
            f"{table_name}: {len(table.columns)} columns, "
            f"{len(table.indexes)} indexes, {len(table.constraints)} constraints"
        )

    click.echo("Core PostgreSQL data contract metadata verified.")


@click.command("verify-project-state")
@with_appcontext
def verify_project_state() -> None:
    """Write a recovery/state realignment summary artifact."""
    summary = build_project_state_summary(write=True, output_path=PROJECT_STATE_SUMMARY_PATH)
    click.echo("Project state realignment")
    click.echo(f"  Proof artifact: {PROJECT_STATE_SUMMARY_PATH}")
    click.echo(
        "  Safe to proceed to Prompt 29: "
        + ("yes" if summary["safe_to_proceed_to_prompt_29"] else "no")
    )
    click.echo(f"  Recommended next action: {summary['recommended_next_action']}")
    click.echo("  Component statuses:")
    for name, component in summary["components"].items():
        click.echo(f"    - {name}: {component['status']}")
