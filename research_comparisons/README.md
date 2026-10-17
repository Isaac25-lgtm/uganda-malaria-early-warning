# `research_comparisons/` — Isolated supervisor-requested comparison studies

> **This directory is NOT part of the frozen main forecasting pipeline, the
> dashboard, the production model-selection workflow, or the dissertation
> results — unless separately approved.**

It holds isolated, supervisor-requested scientific comparison studies that are
kept physically separate from the main application (`app/`) and its canonical
data areas (`data/models`, `data/modelling`, `data/quality`, …).

Rules for every study here:
- **Reads** canonical artifacts **read-only** (the frozen modelling dataset and
  the frozen RF/XGB/SARIMAX/baseline validation predictions/metrics).
- **Writes** only under its own `runs/<RUN_ID>/` directory.
- **Registers nothing** in the Flask application (`app/cli.py` is never touched)
  and never runs any production pipeline, model selection, or forecast
  generation; never writes database rows; never modifies any canonical artifact.
- Runs as a **standalone Python module**, e.g.
  `python -m research_comparisons.literature_based_forecasting_benchmark.src.run --run-id 20260828`.

Deleting this directory leaves the main thesis work fully intact.

## Studies
- `literature_based_forecasting_benchmark/` — do selected literature
  malaria-forecasting methodologies outperform the frozen current Uganda
  district-week models under the same leakage-controlled rolling-origin
  evaluation? (Not an attempt to prove Random Forest is best.)
