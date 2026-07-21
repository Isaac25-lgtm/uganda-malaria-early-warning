"""Population denominator CLI commands."""
from __future__ import annotations

import click
from flask.cli import with_appcontext

from app.services.population import PopulationError, apply_population_to_features


@click.group("population")
def population_cli() -> None:
    """Apply approved UBOS denominators to district-week features."""


@population_cli.command("apply-to-features")
@click.option(
    "--year",
    type=int,
    default=None,
    help="Same-year UBOS denominator application, e.g. 2024 -> epi_year 2024.",
)
@click.option(
    "--use-year",
    type=int,
    default=None,
    help="Explicit denominator year to use for one or more target epi years.",
)
@click.option(
    "--apply-to-years",
    type=str,
    default=None,
    help="Explicit target epi years, e.g. 2020-2025 or 2020,2021,2022.",
)
@click.option("--dry-run", is_flag=True, help="Summarize updates without writing.")
@with_appcontext
def apply_to_features_command(
    year: int | None,
    use_year: int | None,
    apply_to_years: str | None,
    dry_run: bool,
) -> None:
    """Apply UBOS population and incidence_per_1000 for one epi year."""
    try:
        target_years = _parse_year_list(apply_to_years) if apply_to_years else None
        if apply_to_years and use_year is None:
            raise PopulationError("--apply-to-years requires --use-year.")
        if year is None and use_year is None:
            raise PopulationError("Use --year or --use-year.")
        if year is not None and (use_year is not None or apply_to_years is not None):
            raise PopulationError("Use --year for same-year mode, or --use-year with --apply-to-years.")
        summary = apply_population_to_features(
            year,
            use_year=use_year,
            apply_to_years=target_years,
            dry_run=dry_run,
        )
    except PopulationError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("UBOS population denominator application")
    click.echo(f"  Denominator year: {summary.denominator_year}")
    click.echo(f"  Target epi years: {_format_years(summary.target_years)}")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Denominators available: {summary.denominator_count}")
    click.echo(f"  district_week_features rows considered: {summary.feature_rows_considered}")
    click.echo(f"  Rows to update: {summary.rows_to_update}")
    click.echo(f"  Rows updated: {summary.rows_updated}")
    click.echo(f"  Rows unchanged: {summary.rows_unchanged}")
    click.echo(f"  Rows missing denominator: {summary.missing_denominator_rows}")
    click.echo(f"  Incidence values computed: {summary.incidence_computed}")
    click.echo(f"  Incidence left null: {summary.incidence_left_null}")
    if summary.fixed_denominator_assumption:
        click.echo("  Fixed denominator assumption:")
        click.echo(f"    {summary.fixed_denominator_assumption}")

    if summary.districts_missing_denominator:
        click.echo("  District IDs missing denominator:")
        for district_id in sorted(summary.districts_missing_denominator)[:20]:
            click.echo(f"    - {district_id}")
        if len(summary.districts_missing_denominator) > 20:
            click.echo(f"    ... {len(summary.districts_missing_denominator) - 20} more")

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


def _parse_year_list(value: str) -> list[int]:
    cleaned = value.strip()
    if "-" in cleaned and "," not in cleaned:
        start_text, end_text = cleaned.split("-", 1)
        start, end = int(start_text), int(end_text)
        if start > end:
            raise PopulationError("--apply-to-years range start must be <= end.")
        return list(range(start, end + 1))
    years = [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    if not years:
        raise PopulationError("--apply-to-years must contain at least one year.")
    return years


def _format_years(years: list[int]) -> str:
    if not years:
        return "none"
    if years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}-{years[-1]}"
    return ", ".join(str(year) for year in years)
