"""Safe DHIS2 metadata verification client.

This client verifies credentials and metadata before extraction work begins. It
does not write DHIS2 data to PostgreSQL and never logs or stores credentials.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRIES = 2
EXPECTED_MALARIA_CODE = "033B-CD01a"
EXPECTED_MALARIA_NAME = "Malaria (Confirmed) - Cases"


class DHIS2ClientError(RuntimeError):
    """Base class for safe DHIS2 client errors."""


class DHIS2ConfigError(DHIS2ClientError):
    """Raised when required DHIS2 configuration is missing or invalid."""


class DHIS2AuthError(DHIS2ClientError):
    """Raised when DHIS2 rejects credentials."""


class DHIS2TimeoutError(DHIS2ClientError):
    """Raised when DHIS2 does not respond before timeout."""


class DHIS2ConnectionError(DHIS2ClientError):
    """Raised when DHIS2 cannot be reached."""


class DHIS2HTTPError(DHIS2ClientError):
    """Raised for non-200 DHIS2 responses."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DHIS2ResponseFormatError(DHIS2ClientError):
    """Raised when DHIS2 returns unexpected response content."""


@dataclass(frozen=True)
class DHIS2Config:
    base_url: str
    username: str
    password: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    verify_ssl: bool = True


def normalize_base_url(base_url: str) -> str:
    """Return a credential-safe HTTPS base URL for DHIS2 API requests."""
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise DHIS2ConfigError("DHIS2_BASE_URL is required.")
    if any(character.isspace() for character in cleaned):
        raise DHIS2ConfigError("DHIS2_BASE_URL must not contain whitespace.")

    parsed = urlsplit(cleaned)
    if parsed.scheme.lower() != "https":
        raise DHIS2ConfigError(
            "DHIS2_BASE_URL must use https:// because login credentials are "
            "sent to DHIS2 for verification."
        )
    if not parsed.hostname:
        raise DHIS2ConfigError("DHIS2_BASE_URL must include a valid host name.")
    if parsed.username is not None or parsed.password is not None:
        raise DHIS2ConfigError(
            "DHIS2_BASE_URL must not contain embedded credentials."
        )
    if parsed.query or parsed.fragment:
        raise DHIS2ConfigError(
            "DHIS2_BASE_URL must not contain a query string or fragment."
        )
    return cleaned


def parse_bool_env(
    value: str | None,
    *,
    default: bool = True,
    name: str = "DHIS2_VERIFY_SSL",
) -> bool:
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise DHIS2ConfigError(f"{name} must be true or false.")


def format_week_period(year: int, week: int) -> str:
    """Return DHIS2 weekly period format, e.g. 2025W01 or 2025W20."""
    if week < 1 or week > 53:
        raise DHIS2ConfigError("Week must be between 1 and 53.")
    return f"{year}W{week:02d}"


def load_dhis2_config(username: str, password: str) -> DHIS2Config:
    """Build runtime DHIS2 config without reading credentials from a file."""
    base_url = os.environ.get("DHIS2_BASE_URL", "")
    timeout_raw = os.environ.get("DHIS2_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))

    missing = []
    if not base_url.strip():
        missing.append("DHIS2_BASE_URL")
    if not username.strip():
        missing.append("DHIS2 username")
    if not password:
        missing.append("DHIS2 password")
    if missing:
        raise DHIS2ConfigError(
            "Missing DHIS2 configuration: "
            + ", ".join(missing)
            + ". Enter credentials at sign-in or the hidden CLI prompts."
        )

    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise DHIS2ConfigError("DHIS2_TIMEOUT_SECONDS must be numeric.") from exc
    if timeout_seconds <= 0:
        raise DHIS2ConfigError("DHIS2_TIMEOUT_SECONDS must be greater than zero.")

    verify_ssl = parse_bool_env(
        os.environ.get("DHIS2_VERIFY_SSL"),
        default=True,
        name="DHIS2_VERIFY_SSL",
    )
    if not verify_ssl:
        raise DHIS2ConfigError(
            "DHIS2_VERIFY_SSL must remain true for credential-bearing requests."
        )

    return DHIS2Config(
        base_url=normalize_base_url(base_url),
        username=username.strip(),
        password=password,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
    )


class DHIS2Client:
    """Small requests.Session-based DHIS2 client for safe metadata checks."""

    def __init__(
        self,
        config: DHIS2Config,
        *,
        session: requests.Session | None = None,
        retries: int = DEFAULT_RETRIES,
        retry_sleep_seconds: float = 0.2,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.session.auth = (config.username, config.password)
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds

    @classmethod
    def from_credentials(cls, username: str, password: str) -> "DHIS2Client":
        """Create a client from credentials supplied for this operation only."""
        return cls(load_dhis2_config(username, password))

    def close(self) -> None:
        """Drop credential references and close the underlying HTTP session."""
        self.session.auth = None
        close = getattr(self.session, "close", None)
        if callable(close):
            close()
        self.config = DHIS2Config(
            base_url=self.config.base_url,
            username="",
            password="",
            timeout_seconds=self.config.timeout_seconds,
            verify_ssl=self.config.verify_ssl,
        )

    def __enter__(self) -> "DHIS2Client":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def get(self, path: str, *, params: Any | None = None) -> dict[str, Any]:
        """Perform a safe DHIS2 GET and parse JSON."""
        url = self._url(path)
        last_response: requests.Response | None = None

        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_ssl,
                )
            except requests.Timeout as exc:
                raise DHIS2TimeoutError(
                    f"Timed out connecting to DHIS2 endpoint {path!r}."
                ) from exc
            except requests.ConnectionError as exc:
                raise DHIS2ConnectionError(
                    f"Could not connect to DHIS2 endpoint {path!r}."
                ) from exc
            except requests.RequestException as exc:
                raise DHIS2ConnectionError(
                    f"Request to DHIS2 endpoint {path!r} failed."
                ) from exc

            last_response = response
            if response.status_code in TRANSIENT_STATUS_CODES and attempt < self.retries:
                if self.retry_sleep_seconds:
                    time.sleep(self.retry_sleep_seconds)
                continue
            break

        if last_response is None:
            raise DHIS2ConnectionError(f"No response from DHIS2 endpoint {path!r}.")

        if last_response.status_code in {401, 403}:
            raise DHIS2AuthError(
                f"DHIS2 authentication failed for endpoint {path!r} "
                f"(HTTP {last_response.status_code})."
            )
        if last_response.status_code != 200:
            raise DHIS2HTTPError(
                f"DHIS2 endpoint {path!r} returned HTTP {last_response.status_code}.",
                status_code=last_response.status_code,
            )

        try:
            payload = last_response.json()
        except ValueError as exc:
            raise DHIS2ResponseFormatError(
                f"DHIS2 endpoint {path!r} did not return valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise DHIS2ResponseFormatError(
                f"DHIS2 endpoint {path!r} returned JSON that is not an object."
            )
        return payload

    def verify_login(self) -> dict[str, Any]:
        return self.get("/api/me")

    def fetch_metadata_object(self, uid: str) -> tuple[str, dict[str, Any]]:
        """Fetch a data element, indicator or program indicator by UID."""
        fields = (
            "id,code,name,displayName,valueType,aggregationType,domainType,"
            "categoryCombo[id,name,displayName],categoryOptionCombos[id,name,displayName]"
        )
        for resource_type, path in (
            ("dataElement", f"/api/dataElements/{uid}.json"),
            ("indicator", f"/api/indicators/{uid}.json"),
            ("programIndicator", f"/api/programIndicators/{uid}.json"),
        ):
            try:
                return resource_type, self.get(path, params={"fields": fields})
            except DHIS2HTTPError as exc:
                if exc.status_code == 404:
                    continue
                raise
        raise DHIS2HTTPError(
            f"DHIS2 UID {uid!r} was not found as a data element, indicator "
            "or program indicator.",
            status_code=404,
        )

    def fetch_org_units_at_level(self, level: int) -> list[dict[str, Any]]:
        # Presentation evidence: retrieve DHIS2 district metadata before any case values are mapped.
        payload = self.get(
            "/api/organisationUnits.json",
            params={
                "filter": f"level:eq:{level}",
                "fields": (
                    "id,code,name,displayName,level,path,"
                    "parent[id,name,displayName,level],children[id,name,displayName,level]"
                ),
                "paging": "false",
            },
        )
        org_units = payload.get("organisationUnits")
        if not isinstance(org_units, list):
            raise DHIS2ResponseFormatError(
                "DHIS2 organisation unit response missing organisationUnits list."
            )
        return org_units

    def fetch_org_unit_levels(self) -> list[dict[str, Any]]:
        payload = self.get(
            "/api/organisationUnitLevels.json",
            params={"fields": "id,name,level", "paging": "false"},
        )
        levels = payload.get("organisationUnitLevels")
        if not isinstance(levels, list):
            raise DHIS2ResponseFormatError(
                "DHIS2 organisation unit level response missing organisationUnitLevels list."
            )
        return levels

    def fetch_period_types(self) -> list[dict[str, Any]]:
        payload = self.get("/api/periodTypes")
        period_types = payload.get("periodTypes")
        if not isinstance(period_types, list):
            raise DHIS2ResponseFormatError(
                "DHIS2 periodTypes response missing periodTypes list."
            )
        return period_types

    def analytics_smoke_test(self, *, dx: str, year: int, week: int, level: int) -> dict[str, Any]:
        period = format_week_period(year, week)
        return self.analytics_query(dx=dx, periods=[period], level=level)

    def analytics_query(self, *, dx: str, periods: list[str], level: int) -> dict[str, Any]:
        """Query DHIS2 analytics for one dx, a list of weekly periods and org-unit level."""
        if not periods:
            raise DHIS2ConfigError("At least one DHIS2 period is required.")
        # Presentation evidence: cases are requested for the same organisation-unit level and weeks.
        return self.get(
            "/api/analytics",
            params=[
                ("dimension", f"dx:{dx}"),
                ("dimension", "pe:" + ";".join(periods)),
                ("dimension", f"ou:LEVEL-{level}"),
                ("displayProperty", "NAME"),
                ("skipMeta", "false"),
            ],
        )

    def _url(self, path: str) -> str:
        return urljoin(self.config.base_url + "/", path.lstrip("/"))


def read_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise DHIS2ConfigError(f"{name} is required.")
    return value


def read_district_level_from_env() -> int:
    raw = read_required_env("DHIS2_DISTRICT_LEVEL")
    try:
        level = int(raw)
    except ValueError as exc:
        raise DHIS2ConfigError("DHIS2_DISTRICT_LEVEL must be an integer.") from exc
    if level <= 0:
        raise DHIS2ConfigError("DHIS2_DISTRICT_LEVEL must be greater than zero.")
    return level


def malaria_metadata_warnings(metadata: dict[str, Any]) -> list[str]:
    """Return warnings if metadata does not plausibly match confirmed malaria cases."""
    warnings: list[str] = []
    code = str(metadata.get("code") or "")
    name_parts = [
        str(metadata.get("name") or ""),
        str(metadata.get("displayName") or ""),
    ]
    name_text = " ".join(name_parts).lower()

    if code and code != EXPECTED_MALARIA_CODE:
        warnings.append(
            f"Code {code!r} differs from starting assumption {EXPECTED_MALARIA_CODE!r}."
        )
    if "malaria" not in name_text:
        warnings.append("Name/displayName does not contain 'malaria'.")
    if "confirmed" not in name_text:
        warnings.append("Name/displayName does not contain 'confirmed'.")
    if "case" not in name_text and "cases" not in name_text:
        warnings.append("Name/displayName does not clearly mention cases.")

    warnings.append(
        "Confirm with DHIS2/MoH metadata that this captures the intended "
        "confirmed malaria case definition, including RDT and microscopy where applicable."
    )
    return warnings
