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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

TICK_INTERVAL_SECONDS = 2.0  # wall-clock pace of the demo (simulated clock still advances 30s/tick)
N_DEVICES = 14


async def _simulation_loop(app: FastAPI) -> None:
    rt: RuntimeState = app.state.runtime
    try:
        predictor = get_predictor()
        rt.model_available = True
    except FileNotFoundError as e:
        logger.warning("Model not available yet: %s", e)
        predictor = None
        rt.model_available = False

    while True:
        try:
            snapshots = rt.fleet.step()
            rt.ticks += 1
            now = datetime.utcnow()

            for snap in snapshots:
                d = snap.to_dict()
                rt.history[snap.device_id].append(d)
                rt.latest[snap.device_id] = d
                db.insert_metric(d)

                if predictor is None:
                    continue

                window = rt.feature_window(snap.device_id)
                if window is None:
                    continue

                device = rt.devices_by_id[snap.device_id]
                feats = extract_features(window)
                vector = build_model_vector(feats, device.device_type)
                prediction = predictor.predict(vector)
                explanation = explain(prediction, feats, device.name)

                alert = rt.engine.process_tick(asdict(device), prediction.probability, explanation, now)
                if alert is not None:
                    db.upsert_alert(alert.to_dict())

            if rt.ticks % 150 == 0:
                db.prune_old_metrics()

        except Exception:
            logger.exception("simulation tick failed")

        await asyncio.sleep(TICK_INTERVAL_SECONDS)


def _load_persisted_alerts(rt: RuntimeState) -> None:
    for row in db.fetch_all_alerts():
        alert = Alert(
            id=row["id"], device_id=row["device_id"], device_name=row["device_name"],
            device_type=row["device_type"], site=row["site"], category=row["category"],
            severity=row["severity"], probability=row["probability"], eta_minutes=row["eta_minutes"],
            summary=row["summary"], root_cause=row["root_cause"], causes=row["causes"],
            recommended_actions=row["recommended_actions"], status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            detections_count=row["detections_count"], peak_probability=row["peak_probability"],
            feedback=row["feedback"],
            feedback_at=datetime.fromisoformat(row["feedback_at"]) if row["feedback_at"] else None,
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
        )
        rt.engine.load_alert(alert)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    rt = RuntimeState.create(n_devices=N_DEVICES)
    _load_persisted_alerts(rt)
    app.state.runtime = rt
    task = asyncio.create_task(_simulation_loop(app))
    logger.info("Aura-NetGuard simulation loop started (%d devices)", N_DEVICES)
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Aura-NetGuard", lifespan=lifespan)
app.include_router(api_router)

if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
