"""Modelling dataset and validation CLI commands."""
from __future__ import annotations

import click
from flask.cli import with_appcontext

from app.services.modelling_dataset import (
    ModellingDatasetError,
    build_modelling_dataset,
)
from app.services.model_performance import (
    ModelPerformanceError,
    consolidate_model_performance,
)
from app.services.models.baseline import run_baseline_validation
from app.services.models.random_forest import run_random_forest_validation
from app.services.models.sarimax import run_sarimax_validation
from app.services.models.xgboost_model import run_xgboost_validation


@click.group("modelling")
def modelling_cli() -> None:
    """Leakage-safe modelling dataset and validation commands."""


@modelling_cli.command("build-dataset")
@click.option("--start-year", type=int, required=True)
@click.option("--end-year", type=int, required=True)
@click.option("--dry-run", is_flag=True)
@with_appcontext
def build_dataset_command(start_year: int, end_year: int, dry_run: bool) -> None:
    """Build the leakage-safe modelling dataset CSV."""
    try:
        summary = build_modelling_dataset(
            start_year=start_year,
            end_year=end_year,
            dry_run=dry_run,
        )
    except ModellingDatasetError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Modelling dataset build")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Rows: {summary.total_rows}")
    click.echo(f"  Districts represented: {summary.represented_districts}")
    click.echo(f"  Output CSV: {summary.output_path}")
    click.echo(f"  Summary JSON: {summary.summary_path}")
    click.echo(f"  CSV written: {'yes' if summary.csv_written else 'no'}")
    click.echo("  Usable rows by horizon:")
    for horizon, count in summary.usable_rows_by_horizon.items():
        click.echo(f"    - h{horizon}: {count}")
    click.echo("  Leakage checks:")
    for name, passed in summary.leakage_checks.items():
        click.echo(f"    - {name}: {'pass' if passed else 'FAIL'}")


@modelling_cli.command("train-baseline")
@click.option("--horizons", default="4,8,12", show_default=True)
@click.option("--dry-run", is_flag=True)
@with_appcontext
def train_baseline_command(horizons: str, dry_run: bool) -> None:
    """Validate Baseline/Naive model using rolling-origin folds."""
    _run_validation_command(
        runner=run_baseline_validation,
        horizons=horizons,
        dry_run=dry_run,
        label="Baseline/Naive validation",
    )


@modelling_cli.command("train-sarimax")
@click.option("--horizons", default="4,8,12", show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--replace",
    is_flag=True,
    help="Ignore matching district-horizon checkpoints and refit requested jobs.",
)
@with_appcontext
def train_sarimax_command(horizons: str, dry_run: bool, replace: bool) -> None:
    """Validate district-specific SARIMAX models with rolling-origin folds."""
    def report_progress(item) -> None:
        reused = " checkpoint" if item.get("checkpoint_reused") else ""
        click.echo(
            f"  [{item['completed']}/{item['total']}] "
            f"{item['district_id']} h{item['horizon']}: {item['status']}{reused}"
        )

    _run_validation_command(
        runner=run_sarimax_validation,
        horizons=horizons,
        dry_run=dry_run,
        label="SARIMAX validation",
        runner_kwargs={
            "replace": replace,
            "progress_callback": report_progress,
        },
    )


@modelling_cli.command("train-random-forest")
@click.option("--horizons", default="4,8,12", show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--replace",
    is_flag=True,
    help="Ignore matching horizon-fold checkpoints and refit requested jobs.",
)
@with_appcontext
def train_random_forest_command(horizons: str, dry_run: bool, replace: bool) -> None:
    """Validate national-pooled Random Forest models using rolling origins."""
    def report_progress(item) -> None:
        reused = " checkpoint" if item.get("checkpoint_reused") else ""
        click.echo(
            f"  [{item['completed']}/{item['total']}] {item['model']} "
            f"h{item['horizon']} fold {item['validation_fold']}: "
            f"{item['status']}{reused}"
        )

    _run_validation_command(
        runner=run_random_forest_validation,
        horizons=horizons,
        dry_run=dry_run,
        label="Random Forest validation",
        runner_kwargs={
            "replace": replace,
            "progress_callback": report_progress,
        },
    )


@modelling_cli.command("train-xgboost")
@click.option("--horizons", default="4,8,12", show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--replace",
    is_flag=True,
    help="Ignore matching horizon-fold checkpoints and refit requested jobs.",
)
@with_appcontext
def train_xgboost_command(horizons: str, dry_run: bool, replace: bool) -> None:
    """Validate national-pooled XGBoost models using rolling origins."""
    def report_progress(item) -> None:
        reused = " checkpoint" if item.get("checkpoint_reused") else ""
        click.echo(
            f"  [{item['completed']}/{item['total']}] {item['model']} "
            f"h{item['horizon']} fold {item['validation_fold']}: "
            f"{item['status']}{reused}"
        )

    _run_validation_command(
        runner=run_xgboost_validation,
        horizons=horizons,
        dry_run=dry_run,
        label="XGBoost validation",
        runner_kwargs={
            "replace": replace,
            "progress_callback": report_progress,
        },
    )


@modelling_cli.command("consolidate-performance")
@click.option("--dry-run", is_flag=True)
@with_appcontext
def consolidate_performance_command(dry_run: bool) -> None:
    """Consolidate model validation artifacts into model_performance."""
    try:
        summary = consolidate_model_performance(dry_run=dry_run)
    except ModelPerformanceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Model performance consolidation")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Status: {summary.status}")
    click.echo(f"  Records: {len(summary.records)}")
    click.echo(f"  Inserted rows: {summary.inserted_rows}")
    click.echo(f"  Updated rows: {summary.updated_rows}")
    click.echo(f"  Proof artifact: {summary.proof_path}")
    click.echo("  Selected count models:")
    for horizon, selected in summary.selected_count_models.items():
        label = selected["model_name"] if selected else "unavailable"
        click.echo(f"    - h{horizon}: {label}")
    click.echo("  Selected surge models:")
    for horizon, selected in summary.selected_surge_models.items():
        label = selected["model_name"] if selected else "unavailable"
        click.echo(f"    - h{horizon}: {label}")


def _run_validation_command(
    *, runner, horizons: str, dry_run: bool, label: str, runner_kwargs=None
) -> None:
    try:
        selected_horizons = tuple(int(item.strip()) for item in horizons.split(",") if item.strip())
        summary = runner(
            horizons=selected_horizons,
            dry_run=dry_run,
            **(runner_kwargs or {}),
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(label)
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Status: {summary.status}")
    click.echo(f"  Model: {summary.model_name}")
    click.echo(f"  Model scope: {summary.model_scope}")
    click.echo(f"  Horizons: {', '.join(str(item) for item in summary.horizons)}")
    click.echo(f"  Rows: {summary.row_counts.get('modelling_rows', 0)}")
    if summary.count_metrics_by_horizon:
        click.echo("  Count metrics:")
        for horizon, metrics in summary.count_metrics_by_horizon.items():
            click.echo(f"    - h{horizon}: {metrics}")
    if summary.surge_metrics_by_horizon:
        click.echo("  Surge metrics:")
        for horizon, metrics in summary.surge_metrics_by_horizon.items():
            click.echo(f"    - h{horizon}: {metrics}")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
    if getattr(summary, "job_counts", None):
        click.echo(f"  Jobs: {summary.job_counts}")
    if getattr(summary, "artifact_paths", None):
        click.echo(f"  Outputs published: {'yes' if summary.published else 'no'}")
        for name, path in summary.artifact_paths.items():
            click.echo(f"    - {name}: {path}")
