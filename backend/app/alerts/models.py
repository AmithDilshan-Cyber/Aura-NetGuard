from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Probability thresholds, highest first. Below the lowest, no alert is raised
# at all -- silence is the correct output for a healthy device.
SEVERITY_THRESHOLDS = [("CRITICAL", 0.85), ("HIGH", 0.65), ("MEDIUM", 0.45), ("LOW", 0.28)]


def severity_for(probability: float) -> Optional[str]:
    for severity, threshold in SEVERITY_THRESHOLDS:
        if probability >= threshold:
            return severity
    return None


def severity_rank(severity: str) -> int:
    return SEVERITIES.index(severity)


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
    status: str  # open | acknowledged | resolved
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
            if isinstance(d[key], datetime):
                d[key] = d[key].isoformat()
        return d
