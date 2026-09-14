from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .. import db
from ..state import RuntimeState

router = APIRouter(prefix="/api")


class FeedbackRequest(BaseModel):
    is_true_positive: bool


def _runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


@router.get("/devices")
def list_devices(request: Request):
    runtime = _runtime(request)
    return [
        {**asdict(device), "latest": runtime.latest.get(device_id)}
        for device_id, device in runtime.devices_by_id.items()
    ]


@router.get("/devices/{device_id}/metrics")
def device_metrics(device_id: str, request: Request, limit: int = 120):
    runtime = _runtime(request)
    if device_id not in runtime.devices_by_id:
        raise HTTPException(404, "unknown device")
    return list(runtime.history[device_id])[-limit:]


@router.get("/alerts")
def list_alerts(request: Request, status: Optional[str] = None):
    return [a.to_dict() for a in _runtime(request).engine.list_alerts(status=status)]


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, request: Request):
    engine = _runtime(request).engine
    if alert_id not in engine.alerts:
        raise HTTPException(404, "unknown alert")
    alert = engine.acknowledge(alert_id, datetime.utcnow())
    db.upsert_alert(alert.to_dict())
    return alert.to_dict()


@router.post("/alerts/{alert_id}/feedback")
def alert_feedback(alert_id: str, body: FeedbackRequest, request: Request):
    engine = _runtime(request).engine
    if alert_id not in engine.alerts:
        raise HTTPException(404, "unknown alert")
    alert = engine.give_feedback(alert_id, body.is_true_positive, datetime.utcnow())
    db.upsert_alert(alert.to_dict())
    return alert.to_dict()


@router.get("/stats")
def stats(request: Request):
    runtime = _runtime(request)

    state_counts: dict[str, int] = {}
    for device_id in runtime.devices_by_id:
        latest = runtime.latest.get(device_id)
        state = latest["device_state"] if latest else "unknown"
        state_counts[state] = state_counts.get(state, 0) + 1

    active = runtime.engine.active_alerts()
    severity_counts: dict[str, int] = {}
    for alert in active:
        severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

    return {
        "n_devices": len(runtime.devices_by_id),
        "device_state_counts": state_counts,
        "active_alerts": len(active),
        "severity_counts": severity_counts,
        "alerts_suppressed_total": runtime.engine.suppressed_total,
        "feedback_stats": db.feedback_stats(),
        "model_available": runtime.model_available,
        "ticks": runtime.ticks,
    }
