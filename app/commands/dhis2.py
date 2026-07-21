"""DHIS2 metadata verification CLI commands."""
from __future__ import annotations

import json

import click
from pathlib import Path
from flask.cli import with_appcontext

from app.models import DistrictReference
from app.services.dhis2_client import (
    DHIS2Client,
    DHIS2ClientError,
    format_week_period,
    malaria_metadata_warnings,
    read_district_level_from_env,
    read_required_env,
)
from app.services.dhis2_extraction import (
    DHIS2ExtractionError,
    build_extraction_plan,
    extract_malaria_raw,
)
from app.services.reporting_completeness import (
    FALLBACK_STATEMENT,
    REPORTING_COMPLETENESS_ENV,
    ReportingCompletenessError,
    extract_reporting_completeness,
    search_reporting_metrics,
    verify_reporting_metric,
)
from app.services.dhis2_mapping import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_MAPPING_REVIEW_PATH,
    DHIS2MappingError,
    check_week_53_schema,
    inspect_orgunit_levels,
    map_district_orgunits,
    verify_week_period,
)
from app.services.reference_data import (
    DEFAULT_METADATA_PATH,
    EXPECTED_ACTIVE_DISTRICTS,
    read_frame_metadata,
)


@click.group("dhis2")
def dhis2_cli() -> None:
    """DHIS2 connection and metadata verification commands."""


def _interactive_dhis2_client() -> DHIS2Client:
    """Prompt for operation-scoped credentials and close the client afterward."""
    username = click.prompt("DHIS2 username").strip()
    password = click.prompt("DHIS2 password", hide_input=True)
    try:
        client = DHIS2Client.from_credentials(username, password)
    finally:
        password = ""
    click.get_current_context().call_on_close(client.close)
    return client


@dhis2_cli.command("verify-login")
@with_appcontext
def verify_login_command() -> None:
    """Verify DHIS2 credentials using GET /api/me."""
    try:
        client = _interactive_dhis2_client()
        me = client.verify_login()
    except DHIS2ClientError as exc:
        raise click.ClickException(str(exc)) from exc

    display = me.get("displayName") or me.get("name") or me.get("username") or me.get("id")
    click.echo("DHIS2 login verified.")
    click.echo(f"  User: {display or 'unknown'}")
    click.echo("  Password: not displayed or stored")


@dhis2_cli.command("verify-malaria-element")
@with_appcontext
def verify_malaria_element_command() -> None:
    """Verify the configured confirmed malaria data element/indicator UID."""
    try:
        uid = read_required_env("DHIS2_CONFIRMED_MALARIA_DX")
        client = _interactive_dhis2_client()
        resource_type, metadata = client.fetch_metadata_object(uid)
    except DHIS2ClientError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 malaria metadata verification")
    click.echo(f"  Resource type: {resource_type}")
    click.echo(f"  UID: {metadata.get('id') or uid}")
    click.echo(f"  Code: {metadata.get('code') or 'not provided'}")
    click.echo(f"  Name: {metadata.get('name') or 'not provided'}")
    click.echo(f"  Display name: {metadata.get('displayName') or 'not provided'}")
    click.echo(f"  Value type: {metadata.get('valueType') or 'not provided'}")

    warnings = malaria_metadata_warnings(metadata)
    if warnings:
        click.echo("  Warnings:")
        for warning in warnings:
            click.echo(f"    - {warning}")


@dhis2_cli.command("verify-orgunit-level")
@with_appcontext
def verify_orgunit_level_command() -> None:
    """Verify configured DHIS2 district org-unit level."""
    try:
        level = read_district_level_from_env()
        client = _interactive_dhis2_client()
        org_units = client.fetch_org_units_at_level(level)
    except DHIS2ClientError as exc:
        raise click.ClickException(str(exc)) from exc

    metadata = read_frame_metadata(DEFAULT_METADATA_PATH)
    expected_count = metadata.get("expected_active_districts") or EXPECTED_ACTIVE_DISTRICTS
    click.echo("DHIS2 org-unit level verification")
    click.echo(f"  Level: {level}")
    click.echo(f"  Org units returned: {len(org_units)}")
    click.echo(f"  Expected active district frame: {expected_count}")
    if len(org_units) != expected_count:
        click.echo(
            "  Warning: org-unit count does not match the expected analysis frame. "
            "Do not assume this level is districts until verified."
        )

    click.echo("  Sample org units:")
    for org_unit in org_units[:10]:
        name = org_unit.get("displayName") or org_unit.get("name") or "unnamed"
        click.echo(f"    - {name} ({org_unit.get('id')})")

    _compare_org_units_to_reference(org_units)


@dhis2_cli.command("analytics-smoke-test")
@click.option("--year", type=int, required=True, help="Epidemiological year, e.g. 2025.")
@click.option("--week", type=int, required=True, help="Epidemiological week, e.g. 20.")
@with_appcontext
def analytics_smoke_test_command(year: int, week: int) -> None:
    """Run a safe metadata-only /api/analytics smoke test."""
    try:
        dx = read_required_env("DHIS2_CONFIRMED_MALARIA_DX")
        level = read_district_level_from_env()
        period = format_week_period(year, week)
        client = _interactive_dhis2_client()
        payload = client.analytics_smoke_test(dx=dx, year=year, week=week, level=level)
    except DHIS2ClientError as exc:
        raise click.ClickException(str(exc)) from exc

    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise click.ClickException("Analytics response did not contain a rows list.")

    click.echo("DHIS2 analytics smoke test")
    click.echo(f"  dx: {dx}")
    click.echo(f"  pe: {period}")
    click.echo(f"  ou: LEVEL-{level}")
    click.echo(f"  Row count: {len(rows)}")
    if not rows:
        click.echo("  No rows returned. Missing analytics rows must not be treated as zero.")
        return

    click.echo("  First rows:")
    for row in rows[:5]:
        click.echo(f"    - {row}")
    click.echo("  No rows were inserted into raw_dhis2_malaria.")


@dhis2_cli.command("inspect-orgunit-levels")
@with_appcontext
def inspect_orgunit_levels_command() -> None:
    """Inspect DHIS2 org-unit levels and counts."""
    try:
        requested_level = read_district_level_from_env()
        client = _interactive_dhis2_client()
        summary = inspect_orgunit_levels(client, requested_level=requested_level)
    except (DHIS2ClientError, DHIS2MappingError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 org-unit level inspection")
    click.echo(f"  Configured district level: {requested_level}")
    click.echo("  Levels:")
    for level_info in summary.levels:
        raw_level = level_info.get("level")
        try:
            level = int(raw_level)
        except (TypeError, ValueError):
            level = None
        count = summary.counts_by_level.get(level, "not checked") if level else "unknown"
        name = level_info.get("displayName") or level_info.get("name") or "unnamed"
        marker = " <-- configured" if level == requested_level else ""
        click.echo(f"    - level {raw_level}: {name}; org units={count}{marker}")


@dhis2_cli.command("map-district-orgunits")
@click.option("--level", type=int, default=None, help="DHIS2 org-unit level to map.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Fetch, match and write the review CSV without updating district_reference.",
)
@click.option(
    "--allow-mapping-changes",
    is_flag=True,
    help="Allow reviewed changes to existing non-empty DHIS2 org-unit IDs.",
)
@click.option(
    "--alias-path",
    type=click.Path(dir_okay=False),
    default=str(DEFAULT_ALIAS_PATH),
    show_default=True,
    help="Reviewed DHIS2-to-reference name alias CSV.",
)
@click.option(
    "--review-csv",
    type=click.Path(dir_okay=False),
    default=str(DEFAULT_MAPPING_REVIEW_PATH),
    show_default=True,
    help="Mapping review CSV output path.",
)
@with_appcontext
def map_district_orgunits_command(
    level: int | None,
    dry_run: bool,
    allow_mapping_changes: bool,
    alias_path: str,
    review_csv: str,
) -> None:
    """Map live DHIS2 district org-unit UIDs into district_reference."""
    try:
        client = _interactive_dhis2_client()
        summary = map_district_orgunits(
            client,
            level=level,
            dry_run=dry_run,
            allow_mapping_changes=allow_mapping_changes,
            alias_path=Path(alias_path),
            review_csv_path=Path(review_csv),
        )
    except (DHIS2ClientError, DHIS2MappingError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 district org-unit mapping")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Level: {summary.level}")
    click.echo(f"  DHIS2 org units returned: {summary.org_units_returned}")
    click.echo(f"  Active district_reference rows: {summary.active_reference_count}")
    click.echo(f"  Matched org units: {summary.matched_count}")
    click.echo(f"  Exact matches: {summary.exact_matches}")
    click.echo(f"  Normalized matches: {summary.normalized_matches}")
    click.echo(f"  Reviewed alias matches: {summary.alias_matches}")
    click.echo(f"  One-to-one bijection passed: {'yes' if summary.one_to_one_passed else 'no'}")
    click.echo(f"  Rows updated: {summary.rows_updated}")
    click.echo(f"  Review CSV: {summary.review_csv_path}")

    if summary.mapping_changes:
        click.echo("  Mapping changes:")
        for message in summary.mapping_changes[:20]:
            click.echo(f"    - {message}")
        if len(summary.mapping_changes) > 20:
            click.echo(f"    ... {len(summary.mapping_changes) - 20} more")
    if summary.unmatched_dhis2_org_units:
        click.echo("  Unmatched DHIS2 org units:")
        for message in summary.unmatched_dhis2_org_units[:20]:
            click.echo(f"    - {message}")
        if len(summary.unmatched_dhis2_org_units) > 20:
            click.echo(f"    ... {len(summary.unmatched_dhis2_org_units) - 20} more")
    if summary.unmatched_districts:
        click.echo("  Unmatched district_reference rows:")
        for message in summary.unmatched_districts[:20]:
            click.echo(f"    - {message}")
        if len(summary.unmatched_districts) > 20:
            click.echo(f"    ... {len(summary.unmatched_districts) - 20} more")
    if summary.duplicate_dhis2_uids:
        click.echo("  Duplicate DHIS2 UIDs: " + ", ".join(summary.duplicate_dhis2_uids))
    if summary.duplicate_dhis2_names:
        click.echo("  Duplicate DHIS2 normalized names: " + ", ".join(summary.duplicate_dhis2_names))
    if summary.duplicate_matched_district_ids:
        click.echo(
            "  Duplicate matched district_id values: "
            + ", ".join(summary.duplicate_matched_district_ids)
        )
    if summary.duplicate_matched_orgunit_ids:
        click.echo(
            "  Duplicate matched DHIS2 UID values: "
            + ", ".join(summary.duplicate_matched_orgunit_ids)
        )
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


@dhis2_cli.command("verify-week-period")
@click.option("--period", required=True, help="DHIS2 weekly period, e.g. 2025W20.")
@click.option(
    "--no-metadata-update",
    is_flag=True,
    help="Do not update district_frame_metadata.yml with the verification status.",
)
@with_appcontext
def verify_week_period_command(period: str, no_metadata_update: bool) -> None:
    """Verify one DHIS2 weekly period against app ISO week dates."""
    try:
        client = _interactive_dhis2_client()
        summary = verify_week_period(
            client, period=period, update_metadata=not no_metadata_update
        )
    except (DHIS2ClientError, DHIS2MappingError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 weekly period verification")
    click.echo(f"  Period: {summary.period}")
    click.echo(f"  App ISO week start: {summary.app_start_date}")
    click.echo(f"  App ISO week end: {summary.app_end_date}")
    click.echo(f"  DHIS2 start date: {summary.dhis2_start_date or 'not exposed'}")
    click.echo(f"  DHIS2 end date: {summary.dhis2_end_date or 'not exposed'}")
    click.echo(f"  DHIS2 Weekly isoFormat: {summary.weekly_iso_format or 'not exposed'}")
    click.echo(f"  DHIS2 Weekly isoDuration: {summary.weekly_iso_duration or 'not exposed'}")
    click.echo(f"  Analytics row count: {summary.analytics_row_count}")
    click.echo(f"  Status: {summary.status}")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


@dhis2_cli.command("check-week-schema")
@with_appcontext
def check_week_schema_command() -> None:
    """Verify stored week columns allow week 53."""
    summary = check_week_53_schema()
    click.echo("Week-53 schema consistency")
    click.echo(
        f"  district_week_features epi_week allows 1-53: "
        f"{'yes' if summary.district_week_features_allows_53 else 'no'}"
    )
    click.echo(
        f"  district_week_climate epi_week allows 1-53: "
        f"{'yes' if summary.district_week_climate_allows_53 else 'no'}"
    )
    click.echo(
        f"  raw_dhis2_malaria epi_week allows 1-53: "
        f"{'yes' if summary.raw_dhis2_malaria_allows_53 else 'no'}"
    )
    for detail in summary.details:
        click.echo(f"  {detail}")
    if not summary.passed:
        raise click.ClickException("One or more week columns do not allow epi_week 53.")


@dhis2_cli.command("extract-malaria")
@click.option("--start-year", type=int, default=None, help="Start year for extraction.")
@click.option("--end-year", type=int, default=None, help="End year for extraction.")
@click.option("--year", type=int, default=None, help="Extract one year only.")
@click.option("--start-week", type=int, default=None, help="Start week within selected year(s).")
@click.option("--end-week", type=int, default=None, help="End week within selected year(s).")
@click.option("--week", type=int, default=None, help="Extract one epidemiological week only.")
@click.option(
    "--include-week-53",
    is_flag=True,
    help="Include week 53 when extracting full or ranged years.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print planned requests after prechecks without inserting raw rows.",
)
@with_appcontext
def extract_malaria_command(
    start_year: int | None,
    end_year: int | None,
    year: int | None,
    start_week: int | None,
    end_week: int | None,
    week: int | None,
    include_week_53: bool,
    dry_run: bool,
) -> None:
    """Extract raw weekly confirmed malaria analytics rows from DHIS2."""
    try:
        chunks = build_extraction_plan(
            start_year=start_year,
            end_year=end_year,
            year=year,
            start_week=start_week,
            end_week=end_week,
            week=week,
            include_week_53=include_week_53,
        )
        client = _interactive_dhis2_client()
        summary = extract_malaria_raw(client, chunks, dry_run=dry_run)
    except (DHIS2ClientError, DHIS2ExtractionError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 raw malaria extraction")
    if dry_run:
        click.echo("  Mode: dry-run (no pipeline_run and no raw rows inserted)")
    else:
        click.echo(f"  pipeline_run_id: {summary.pipeline_run_id}")
    click.echo(f"  dx UID: {summary.dx_uid or 'not available'}")
    click.echo(f"  dx object type: {summary.dx_object_type or 'not available'}")
    click.echo(f"  dx name: {summary.dx_name or 'not provided'}")
    click.echo(f"  aggregation type: {summary.aggregation_type or 'not provided'}")
    click.echo(f"  org-unit level: {summary.district_level or 'not available'}")
    click.echo(f"  Periods requested: {len(summary.requested_periods)}")
    click.echo(f"  First periods: {', '.join(summary.requested_periods[:10])}")
    if len(summary.requested_periods) > 10:
        click.echo(f"  ... {len(summary.requested_periods) - 10} more")
    click.echo(f"  Total DHIS2 rows returned: {summary.row_count_returned}")
    click.echo(f"  Rows inserted: {summary.rows_inserted}")
    click.echo(f"  Explicit zero values inserted: {summary.explicit_zero_values}")
    click.echo(f"  Duplicate rows skipped: {summary.duplicate_rows_skipped}")
    click.echo(f"  Mapped DHIS2 org units: {len(summary.mapped_org_units)}")
    click.echo(f"  Unmapped DHIS2 org units: {len(summary.unmapped_org_units)}")
    click.echo(f"  Periods with no returned rows: {len(summary.empty_periods)}")
    click.echo(f"  Chunk fallback used: {'yes' if summary.fallback_used else 'no'}")

    if summary.unmapped_org_units:
        click.echo("  Unmapped org units:")
        for orgunit_id in sorted(summary.unmapped_org_units)[:20]:
            click.echo(f"    - {orgunit_id}")
        if len(summary.unmapped_org_units) > 20:
            click.echo(f"    ... {len(summary.unmapped_org_units) - 20} more")

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


@dhis2_cli.command("search-reporting-metrics")
@click.option(
    "--query",
    default="reporting rate",
    show_default=True,
    help="Metadata search text, e.g. 'reporting rate' or 'completeness'.",
)
@click.option("--max-results", type=int, default=25, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional JSON artifact path for search results.",
)
@with_appcontext
def search_reporting_metrics_command(
    query: str, max_results: int, output: str | None
) -> None:
    """Search DHIS2 metadata for candidate facility reporting completeness metrics."""
    try:
        client = _interactive_dhis2_client()
        summary = search_reporting_metrics(client, query, max_results=max_results)
    except (DHIS2ClientError, ReportingCompletenessError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 reporting-completeness metadata search")
    click.echo(f"  Query: {summary.query}")
    click.echo(f"  Candidates found: {len(summary.candidates)}")
    if summary.candidates:
        click.echo("  Candidates:")
        for candidate in summary.candidates[:20]:
            name = candidate.display_name or candidate.name or "unnamed"
            code = candidate.code or "no code"
            click.echo(
                f"    - {candidate.resource_type} {candidate.uid}: {name} "
                f"[{code}] ({candidate.suitability})"
            )
        if len(summary.candidates) > 20:
            click.echo(f"    ... {len(summary.candidates) - 20} more")
    else:
        click.echo("  No candidates found.")
        click.echo(f"  {FALLBACK_STATEMENT}")

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
    if output:
        _write_json_artifact(Path(output), summary.to_dict())
        click.echo(f"  Search artifact: {output}")


@dhis2_cli.command("verify-reporting-metric")
@click.option(
    "--uid",
    default=None,
    help=f"DHIS2 metric UID. Defaults to ${REPORTING_COMPLETENESS_ENV} when omitted.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional JSON artifact path for verification result.",
)
@with_appcontext
def verify_reporting_metric_command(uid: str | None, output: str | None) -> None:
    """Verify one candidate reporting-completeness metric before extraction."""
    try:
        selected_uid = uid or read_required_env(REPORTING_COMPLETENESS_ENV)
        client = _interactive_dhis2_client()
        verification = verify_reporting_metric(client, selected_uid)
    except (DHIS2ClientError, ReportingCompletenessError) as exc:
        raise click.ClickException(str(exc)) from exc

    candidate = verification.candidate
    click.echo("DHIS2 reporting-completeness metric verification")
    click.echo(f"  UID: {candidate.uid}")
    click.echo(f"  Resource type: {candidate.resource_type}")
    click.echo(f"  Code: {candidate.code or 'not provided'}")
    click.echo(f"  Name: {candidate.name or 'not provided'}")
    click.echo(f"  Display name: {candidate.display_name or 'not provided'}")
    click.echo(f"  Period type: {candidate.period_type or 'not exposed'}")
    click.echo(f"  Extractable as analytics dx: {'yes' if verification.extractable_dx else 'no'}")
    click.echo(f"  Metadata plausibly reporting completeness: {'yes' if verification.metadata_plausible else 'no'}")
    click.echo(f"  Usable for 2025 dry-run gate: {'yes' if verification.usable_for_dry_run else 'no'}")
    if candidate.formula:
        click.echo(f"  Formula text: {candidate.formula[:500]}")
    if verification.warnings:
        click.echo("  Warnings:")
        for warning in verification.warnings:
            click.echo(f"    - {warning}")
    if not verification.usable_for_dry_run:
        click.echo(f"  {FALLBACK_STATEMENT}")
    if output:
        _write_json_artifact(Path(output), verification.to_dict())
        click.echo(f"  Verification artifact: {output}")


@dhis2_cli.command("extract-reporting-completeness")
@click.option("--start-year", type=int, default=None, help="Start year for extraction.")
@click.option("--end-year", type=int, default=None, help="End year for extraction.")
@click.option("--year", type=int, default=None, help="Extract one year only.")
@click.option("--start-week", type=int, default=None, help="Start week within selected year(s).")
@click.option("--end-week", type=int, default=None, help="End week within selected year(s).")
@click.option("--week", type=int, default=None, help="Extract one epidemiological week only.")
@click.option(
    "--include-week-53",
    is_flag=True,
    help="Include week 53 when extracting full or ranged years.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run prechecks and analytics parsing without updating district_week_features.",
)
@with_appcontext
def extract_reporting_completeness_command(
    start_year: int | None,
    end_year: int | None,
    year: int | None,
    start_week: int | None,
    end_week: int | None,
    week: int | None,
    include_week_53: bool,
    dry_run: bool,
) -> None:
    """Extract verified DHIS2 facility reporting completeness into feature rows."""
    try:
        chunks = build_extraction_plan(
            start_year=start_year,
            end_year=end_year,
            year=year,
            start_week=start_week,
            end_week=end_week,
            week=week,
            include_week_53=include_week_53,
        )
        client = _interactive_dhis2_client()
        summary = extract_reporting_completeness(client, chunks, dry_run=dry_run)
    except (DHIS2ClientError, DHIS2ExtractionError, ReportingCompletenessError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 facility reporting completeness extraction")
    if dry_run:
        click.echo("  Mode: dry-run (no pipeline_run and no feature rows updated)")
    else:
        click.echo(f"  pipeline_run_id: {summary.pipeline_run_id}")
    click.echo(f"  dx UID: {summary.dx_uid or 'not available'}")
    click.echo(f"  dx object type: {summary.dx_object_type or 'not available'}")
    click.echo(f"  dx name: {summary.dx_name or 'not provided'}")
    click.echo(f"  org-unit level: {summary.district_level or 'not available'}")
    click.echo(f"  Periods requested: {len(summary.requested_periods)}")
    click.echo(f"  Total DHIS2 rows returned: {summary.row_count_returned}")
    click.echo(f"  Rows normalized to percent: {summary.rows_normalized}")
    click.echo(f"  Existing district-week rows updated: {summary.rows_updated}")
    click.echo(f"  Rows without existing district-week feature: {summary.rows_without_feature}")
    click.echo(f"  Duplicate rows skipped: {summary.duplicate_rows_skipped}")
    click.echo(f"  Mapped DHIS2 org units: {len(summary.mapped_org_units)}")
    click.echo(f"  Unmapped DHIS2 org units: {len(summary.unmapped_org_units)}")
    click.echo(f"  Chunk fallback used: {'yes' if summary.fallback_used else 'no'}")
    click.echo(f"  Min/median/max percent: {_format_decimal(summary.min_value)} / {_format_decimal(summary.median_value)} / {_format_decimal(summary.max_value)}")
    if summary.normalization_rules:
        click.echo("  Normalization rules:")
        for rule, count in sorted(summary.normalization_rules.items()):
            click.echo(f"    - {rule}: {count}")
    if summary.unmapped_org_units:
        click.echo("  Unmapped org units:")
        for orgunit_id in sorted(summary.unmapped_org_units)[:20]:
            click.echo(f"    - {orgunit_id}")
        if len(summary.unmapped_org_units) > 20:
            click.echo(f"    ... {len(summary.unmapped_org_units) - 20} more")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


def _compare_org_units_to_reference(org_units: list[dict]) -> None:
    live_ids = {org_unit.get("id") for org_unit in org_units if org_unit.get("id")}
    mapped_rows = DistrictReference.query.filter(
        DistrictReference.dhis2_orgunit_id.isnot(None),
        DistrictReference.dhis2_orgunit_id != "",
    ).all()
    mapped_ids = {row.dhis2_orgunit_id for row in mapped_rows}

    if not mapped_rows:
        click.echo("  district_reference contains no DHIS2 org-unit mappings yet.")
        return

    found = sorted(mapped_ids & live_ids)
    missing = sorted(mapped_ids - live_ids)
    unmapped_live = sorted(live_ids - mapped_ids)
    click.echo(f"  Mapped org units found in DHIS2: {len(found)}")
    click.echo(f"  Mapped org units missing from DHIS2: {len(missing)}")
    click.echo(f"  DHIS2 org units not mapped in district_reference: {len(unmapped_live)}")

    if missing:
        click.echo("  Missing mapped org units:")
        for orgunit_id in missing[:20]:
            click.echo(f"    - {orgunit_id}")
        if len(missing) > 20:
            click.echo(f"    ... {len(missing) - 20} more")


def _write_json_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_decimal(value) -> str:
    return str(value) if value is not None else "not available"
