from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .alerts.explain import explain
from .alerts.models import Alert
from .api.routes import router as api_router
from .features import build_model_vector, extract_features
from .ml.predictor import get_predictor
from .state import RuntimeState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura_netguard")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

TICK_INTERVAL_SECONDS = 2.0  # wall-clock demo pace; the simulated clock still advances 30s/tick
PRUNE_EVERY_TICKS = 150
N_DEVICES = 14


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _load_persisted_alerts(runtime: RuntimeState) -> None:
    for row in db.fetch_all_alerts():
        runtime.engine.load_alert(
            Alert(
                id=row["id"],
                device_id=row["device_id"],
                device_name=row["device_name"],
                device_type=row["device_type"],
                site=row["site"],
                category=row["category"],
                severity=row["severity"],
                probability=row["probability"],
                eta_minutes=row["eta_minutes"],
                summary=row["summary"],
                root_cause=row["root_cause"],
                causes=row["causes"],
                recommended_actions=row["recommended_actions"],
                status=row["status"],
                created_at=_parse_dt(row["created_at"]),
                updated_at=_parse_dt(row["updated_at"]),
                last_seen_at=_parse_dt(row["last_seen_at"]),
                detections_count=row["detections_count"],
                peak_probability=row["peak_probability"],
                feedback=row["feedback"],
                feedback_at=_parse_dt(row["feedback_at"]),
                resolved_at=_parse_dt(row["resolved_at"]),
            )
        )


def _run_tick(runtime: RuntimeState, predictor) -> None:
    now = datetime.utcnow()
    runtime.ticks += 1

    for snapshot in runtime.fleet.step():
        row = snapshot.to_dict()
        runtime.record(row)
        db.insert_metric(row)

        if predictor is None:
            continue

        window = runtime.feature_window(snapshot.device_id)
        if window is None:
            continue

        device = runtime.devices_by_id[snapshot.device_id]
        feats = extract_features(window)
        prediction = predictor.predict(build_model_vector(feats, device.device_type))
        explanation = explain(prediction, feats, device.name)

        alert = runtime.engine.process_tick(asdict(device), prediction.probability, explanation, now)
        if alert is not None:
            db.upsert_alert(alert.to_dict())

    if runtime.ticks % PRUNE_EVERY_TICKS == 0:
        db.prune_old_metrics()


async def _simulation_loop(app: FastAPI) -> None:
    runtime: RuntimeState = app.state.runtime
    try:
        predictor = get_predictor()
        runtime.model_available = True
    except FileNotFoundError as exc:
        logger.warning("Running without predictions: %s", exc)
        predictor = None

    while True:
        try:
            _run_tick(runtime, predictor)
        except Exception:
            logger.exception("simulation tick failed")
        await asyncio.sleep(TICK_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    runtime = RuntimeState.create(n_devices=N_DEVICES)
    _load_persisted_alerts(runtime)
    app.state.runtime = runtime

    task = asyncio.create_task(_simulation_loop(app))
    logger.info("Aura-NetGuard simulation loop started (%d devices)", N_DEVICES)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Aura-NetGuard", lifespan=lifespan)
app.include_router(api_router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
