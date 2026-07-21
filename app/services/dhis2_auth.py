"""Transient DHIS2-backed authentication for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from flask import current_app, session
from flask_login import UserMixin, login_user, logout_user

from app.extensions import login_manager
from app.services.dhis2_client import DHIS2Client, DHIS2ResponseFormatError

DHIS2_IDENTITY_SESSION_KEY = "dhis2_identity"
DHIS2_IDENTITY_VERSION = 1


@dataclass(frozen=True)
class DHIS2SessionUser(UserMixin):
    """Non-secret DHIS2 identity retained for one browser session."""

    id: str
    display_name: str
    verified_at: int
    role: str = "DHIS2 verified"

    @property
    def username(self) -> str:
        """Compatibility display value for the existing navigation template."""
        return self.display_name

    @property
    def is_admin(self) -> bool:
        return False

    def to_session(self) -> dict[str, Any]:
        return {
            "version": DHIS2_IDENTITY_VERSION,
            "id": self.id,
            "display_name": self.display_name,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_session(cls, payload: Any) -> "DHIS2SessionUser | None":
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != DHIS2_IDENTITY_VERSION:
            return None
        user_id = str(payload.get("id") or "").strip()
        display_name = str(payload.get("display_name") or "").strip()
        try:
            verified_at = int(payload.get("verified_at"))
        except (TypeError, ValueError):
            return None
        if not user_id or not display_name or verified_at <= 0:
            return None
        return cls(id=user_id, display_name=display_name, verified_at=verified_at)


def authenticate_dhis2_user(username: str, password: str) -> DHIS2SessionUser:
    """Verify submitted credentials against live DHIS2 and retain no secret."""
    with DHIS2Client.from_credentials(username, password) as client:
        payload = client.verify_login()

    user_id = str(payload.get("id") or "").strip()
    display_name = str(payload.get("displayName") or payload.get("name") or "").strip()
    if not user_id:
        raise DHIS2ResponseFormatError("DHIS2 /api/me did not return a user id.")
    if not display_name:
        display_name = "DHIS2 user"
    return DHIS2SessionUser(
        id=user_id,
        display_name=display_name,
        verified_at=int(datetime.now(timezone.utc).timestamp()),
    )


def establish_dhis2_session(identity: DHIS2SessionUser) -> None:
    """Create a non-persistent login session containing no credentials."""
    session.clear()
    session.permanent = False
    session[DHIS2_IDENTITY_SESSION_KEY] = identity.to_session()
    login_user(identity, remember=False, fresh=True)


def clear_dhis2_session() -> None:
    logout_user()
    session.clear()


@login_manager.user_loader
def load_dhis2_session_user(user_id: str) -> DHIS2SessionUser | None:
    identity = DHIS2SessionUser.from_session(session.get(DHIS2_IDENTITY_SESSION_KEY))
    if identity is None or identity.get_id() != user_id:
        return None

    max_age_seconds = int(current_app.config["DHIS2_LOGIN_SESSION_MINUTES"]) * 60
    now = int(datetime.now(timezone.utc).timestamp())
    if now - identity.verified_at > max_age_seconds:
        session.clear()
        return None
    return identity
