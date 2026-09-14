from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .. import db

router = APIRouter(prefix="/api")


class FeedbackRequest(BaseModel):
    is_true_positive: bool


def _runtime(request: Request):
    return request.app.state.runtime


@router.get("/devices")
def list_devices(request: Request):
    rt = _runtime(request)
    out = []
    for device_id, device in rt.devices_by_id.items():
        latest = rt.latest.get(device_id)
        out.append({**asdict(device), "latest": latest})
    return out


@router.get("/devices/{device_id}/metrics")
def device_metrics(device_id: str, request: Request, limit: int = 120):
    rt = _runtime(request)
    if device_id not in rt.devices_by_id:
        raise HTTPException(404, "unknown device")
    hist = list(rt.history[device_id])[-limit:]
    return hist


@router.get("/alerts")
def list_alerts(request: Request, status: str | None = None):
    rt = _runtime(request)
    return [a.to_dict() for a in rt.engine.list_alerts(status=status)]


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, request: Request):
    rt = _runtime(request)
    if alert_id not in rt.engine.alerts:
        raise HTTPException(404, "unknown alert")
    alert = rt.engine.acknowledge(alert_id, datetime.utcnow())
    db.upsert_alert(alert.to_dict())
    return alert.to_dict()


@router.post("/alerts/{alert_id}/feedback")
def alert_feedback(alert_id: str, body: FeedbackRequest, request: Request):
    rt = _runtime(request)
    if alert_id not in rt.engine.alerts:
        raise HTTPException(404, "unknown alert")
    alert = rt.engine.give_feedback(alert_id, body.is_true_positive, datetime.utcnow())
    db.upsert_alert(alert.to_dict())
    return alert.to_dict()


@router.get("/stats")
def stats(request: Request):
    rt = _runtime(request)
    devices = list(rt.devices_by_id.values())
    state_counts: dict[str, int] = {}
    for device in devices:
        latest = rt.latest.get(device.device_id)
        s = latest["device_state"] if latest else "unknown"
        state_counts[s] = state_counts.get(s, 0) + 1

    open_alerts = rt.engine.list_alerts(status="open")
    ack_alerts = rt.engine.list_alerts(status="acknowledged")
    severity_counts: dict[str, int] = {}
    for a in open_alerts + ack_alerts:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

    return {
        "n_devices": len(devices),
        "device_state_counts": state_counts,
        "active_alerts": len(open_alerts) + len(ack_alerts),
        "severity_counts": severity_counts,
        "alerts_suppressed_total": rt.engine.suppressed_total,
        "feedback_stats": db.feedback_stats(),
        "model_available": rt.model_available,
        "ticks": rt.ticks,
    }
