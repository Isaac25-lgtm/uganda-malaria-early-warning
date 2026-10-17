"""Isolated, supervisor-requested research comparison studies.

NOT part of the frozen main forecasting pipeline, dashboard, production
model-selection workflow, or dissertation results unless separately approved.
Each sub-package reads canonical artifacts read-only and writes only under its
own runs/<RUN_ID>/ directory. Nothing here registers with the Flask app.
"""
