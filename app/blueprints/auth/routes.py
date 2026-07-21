"""Live DHIS2 authentication routes."""
from urllib.parse import urlsplit

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.blueprints.auth import bp
from app.blueprints.auth.forms import LoginForm
from app.services.dhis2_auth import (
    authenticate_dhis2_user,
    clear_dhis2_session,
    establish_dhis2_session,
)
from app.services.dhis2_client import DHIS2AuthError, DHIS2ClientError


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data
        try:
            identity = authenticate_dhis2_user(username, password)
        except DHIS2AuthError:
            flash("DHIS2 sign-in failed. Check your credentials and access.", "error")
            return redirect(url_for("auth.login"))
        except DHIS2ClientError:
            flash("DHIS2 authentication service is unavailable. Try again shortly.", "error")
            return redirect(url_for("auth.login"))
        finally:
            form.password.data = None
            password = ""

        establish_dhis2_session(identity)

        # Only permit local redirect targets (avoid open-redirect).
        next_page = request.args.get("next")
        parsed = urlsplit(next_page or "")
        if (
            not next_page
            or parsed.scheme
            or parsed.netloc
            or not parsed.path.startswith("/")
            or parsed.path.startswith("//")
        ):
            next_page = url_for("dashboard.index")
        return redirect(next_page)

    return render_template("auth/login.html", form=form)


@bp.route("/logout")
@login_required
def logout():
    clear_dhis2_session()
    flash("You have been signed out.", "success")
    return redirect(url_for("auth.login"))


@bp.after_request
def prevent_auth_response_caching(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response
