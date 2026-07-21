"""Authentication forms (Flask-WTF provides CSRF protection)."""
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Length


class LoginForm(FlaskForm):
    username = StringField(
        "DHIS2 username", validators=[DataRequired(), Length(max=200)]
    )
    password = PasswordField(
        "DHIS2 password", validators=[DataRequired(), Length(max=1024)]
    )
    submit = SubmitField("Verify and sign in")
