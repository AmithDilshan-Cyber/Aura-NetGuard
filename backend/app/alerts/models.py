from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def severity_for(probability: float) -> Optional[str]:
    if probability >= 0.85:
        return "CRITICAL"
    if probability >= 0.65:
        return "HIGH"
    if probability >= 0.45:
        return "MEDIUM"
    if probability >= 0.28:
        return "LOW"
    return None


def severity_rank(sev: str) -> int:
    return SEVERITIES.index(sev)


@dataclass
class CauseDict:
    metric: str
    label: str
    value: float
    unit: str
    contribution: float


@dataclass
class Alert:
    id: str
    device_id: str
    device_name: str
    device_type: str
    site: str
    category: str
    severity: str
    probability: float
    eta_minutes: Optional[float]
    summary: str
    root_cause: str
    causes: list[dict]
    recommended_actions: list[str]
    status: str  # open | acknowledged | resolved | suppressed
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime
    detections_count: int = 1
    peak_probability: float = 0.0
    feedback: Optional[str] = None  # true_positive | false_positive
    feedback_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("created_at", "updated_at", "last_seen_at", "feedback_at", "resolved_at"):
            if d[key] is not None:
                d[key] = d[key].isoformat() if isinstance(d[key], datetime) else d[key]
        return d
