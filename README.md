# Aura-NetGuard

**AI-Based Network Failure Prediction & Human-Centred Explainable Alert System**

A proactive network monitoring prototype that forecasts device-level network
failures *before* they happen (not just after a threshold is breached), and
delivers each prediction as a human-centred alert: a plain-language root
cause, an estimated time-to-failure, concrete remediation steps, and a
feedback loop that lets operators actively suppress repeat false alarms.

Built as an undergraduate research project — see [`RESEARCH.md`](RESEARCH.md)
for the full write-up: problem statement, literature/gap analysis, novel
contributions, architecture, methodology, and evaluation.

## What it does

- Simulates a fleet of network devices (routers, switches, access points,
  firewalls, load balancers) with realistic telemetry and injects three
  fault archetypes: gradual degradation, resource exhaustion, and link flap.
- Extracts rolling-window features (trend, mean, std, slope) from each
  device's recent telemetry and feeds them into a `RandomForestClassifier`
  trained to predict failure **within the next ~7.5 minutes**.
- Explains every prediction using SHAP feature attribution, mapped to a
  plain-language root cause (physical-layer instability, resource
  exhaustion, congestion, or thermal) with recommended remediation steps.
- Runs an alert engine that de-duplicates repeat detections, auto-resolves
  stale alerts, and — when an operator marks an alert a false alarm —
  suppresses that specific device+cause combination for a cooldown period
  (unless risk escalates to CRITICAL).
- Serves a live dashboard: fleet health grid, per-device metric charts, and
  an alert feed with Acknowledge / Confirm real issue / False alarm controls.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# train the failure-prediction model (writes backend/app/ml/model_store/)
python -m backend.app.ml.train

# run the app
uvicorn backend.app.main:app --reload
```

Then open http://localhost:8000 — the dashboard polls the live simulated
fleet and will start surfacing predicted-failure alerts as faults are
randomly injected (usually within the first couple of minutes).

## Tests

```bash
pytest
```

## Project layout

```
backend/app/
  simulator.py        synthetic telemetry generator + fault injection
  features.py          rolling-window feature engineering (shared train/serve)
  ml/
    dataset.py          builds a labelled dataset from the simulator
    train.py             trains + evaluates the model (device-level split)
    predictor.py         loads the model, runs SHAP explainability
  alerts/
    explain.py           SHAP -> root cause, plain-language summary, ETA, actions
    engine.py             severity scoring, de-duplication, cooldown, feedback
    models.py             Alert data model
  db.py                  SQLite persistence for metrics + alerts/feedback
  api/routes.py           REST API
  main.py                 FastAPI app + background simulation loop
frontend/                 dashboard (vanilla HTML/CSS/JS + Chart.js)
backend/tests/            pytest suite
RESEARCH.md               full research write-up
```

## Applying this to a real company network

The simulator sits behind a single interface (`FleetSimulator.step()`
producing metric snapshots), so it can be replaced with a real SNMP,
NetFlow, or Prometheus collector without touching the model, explainer,
alert engine, or dashboard — see §8 of `RESEARCH.md` for the adapter
pattern.
