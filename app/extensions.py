"""Shared Flask extension instances.

Kept separate from the application factory to avoid circular imports: modules
import these objects from here, and ``create_app`` binds them to the app via
``init_app``.
"""
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

# Unauthenticated users hitting a protected route are redirected here.
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to continue."
login_manager.login_message_category = "warning"
