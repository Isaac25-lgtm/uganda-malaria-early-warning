"""Forecast generation CLI commands."""
from __future__ import annotations

import click
from flask.cli import with_appcontext

from app.services.forecast_generation import ForecastGenerationError, generate_forecasts


@click.group("forecasting")
def forecasting_cli() -> None:
    """Forecast output generation commands."""


@forecasting_cli.command("generate")
@click.option("--latest", is_flag=True, help="Use the latest available feature week.")
@click.option("--as-of-year", type=int, default=None)
@click.option("--as-of-week", type=int, default=None)
@click.option("--models", default="baseline,sarimax,random_forest,xgboost", show_default=True)
@click.option("--horizons", default="4,8,12", show_default=True)
@click.option("--dry-run", is_flag=True)
@with_appcontext
def generate_command(
    latest: bool,
    as_of_year: int | None,
    as_of_week: int | None,
    models: str,
    horizons: str,
    dry_run: bool,
) -> None:
    """Fit final production models and generate forecast_outputs."""
    def report_progress(item) -> None:
        district = f" {item['district_id']}" if item.get("district_id") else ""
        click.echo(
            f"  [{item['completed']}/{item['total']}] {item['model']} "
            f"h{item['horizon']}{district}: {item['status']}"
        )

    try:
        selected_models = tuple(item.strip() for item in models.split(",") if item.strip())
        selected_horizons = tuple(int(item.strip()) for item in horizons.split(",") if item.strip())
        summary = generate_forecasts(
            latest=latest,
            as_of_year=as_of_year,
            as_of_week=as_of_week,
            models=selected_models,
            horizons=selected_horizons,
            dry_run=dry_run,
            progress_callback=report_progress,
        )
    except ForecastGenerationError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Forecast generation")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Status: {summary.status}")
    click.echo(f"  Issue week: {summary.issue_year}W{summary.issue_week:02d}")
    click.echo(f"  Composite index confirmed: {'yes' if summary.composite_index_confirmed else 'no'}")
    click.echo(f"  Production fitting ready: {'yes' if summary.production_fitting_ready else 'no'}")
    click.echo(f"  Expected rows: {summary.expected_forecast_rows}")
    click.echo(f"  Generated rows: {summary.generated_forecast_rows}")
    click.echo(f"  Inserted rows: {summary.inserted_rows}")
    click.echo(f"  Updated rows: {summary.updated_rows}")
    click.echo(f"  Final issue row count: {summary.final_issue_row_count}")
    click.echo(f"  Published: {'yes' if summary.published else 'no'}")
    click.echo("  Expected model fits:")
    for model, count in summary.expected_model_fits.items():
        click.echo(f"    - {model}: {count}")
    if summary.fit_counts:
        click.echo(f"  Fit outcomes: {summary.fit_counts}")
    click.echo(f"  Target weeks:")
    for horizon, target in summary.target_weeks.items():
        click.echo(f"    - h{horizon}: {target}")
    if summary.models_skipped:
        click.echo("  Models skipped:")
        for model, reason in summary.models_skipped.items():
            click.echo(f"    - {model}: {reason}")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
    click.echo("  Output paths:")
    for name, path in summary.artifact_paths.items():
        click.echo(f"    - {name}: {path}")
