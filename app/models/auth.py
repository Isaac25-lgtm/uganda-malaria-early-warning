"""Legacy local-user table retained for schema compatibility.

Dashboard authentication is performed against live DHIS2. Existing local user
rows and password hashes are not consulted by any login route.
"""
from __future__ import annotations

from app.extensions import db

# Simple role set for now. Kept intentionally minimal.
ROLE_ADMIN = "admin"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_ADMIN, ROLE_VIEWER)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(
        db.String(20), nullable=False, default=ROLE_VIEWER, server_default=ROLE_VIEWER
    )
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, server_default=db.func.now()
    )

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.username!r} role={self.role!r}>"
