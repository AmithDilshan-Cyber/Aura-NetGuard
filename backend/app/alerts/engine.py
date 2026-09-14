"""Stateful, human-in-the-loop alert engine.

Two behaviours here exist specifically to counter alert fatigue, which the
literature consistently identifies as the reason predictive network alerts
get ignored in practice:

  1. De-duplication: repeated positive predictions for the same
     (device, root-cause-category) update a single open alert instead of
     spawning a new one every 30-second tick.

  2. Feedback-driven cooldown: when an operator marks an alert a false
     positive, that exact (device, category) combination is suppressed for
     a cooldown window -- unless the risk escalates to CRITICAL, which
     always breaks through. This makes the trust the system earns from an
     operator's feedback have a direct, visible effect on future noise,
     which is the human-centred contribution this project evaluates.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

from .explain import Explanation
from .models import Alert, severity_for

AUTO_RESOLVE_STREAK = 4  # consecutive low-risk ticks before an open alert auto-resolves
COOLDOWN_MINUTES = 20

ACTIVE_STATUSES = ("open", "acknowledged")


class AlertEngine:
    def __init__(self) -> None:
        self.alerts: dict[str, Alert] = {}
        self.suppressed_total = 0
        self._active_index: dict[tuple[str, str], str] = {}
        self._low_streak: dict[tuple[str, str], int] = {}
        self._cooldown_until: dict[tuple[str, str], datetime] = {}

    def load_alert(self, alert: Alert) -> None:
        """Rehydrate an alert (e.g. from persistent storage) on startup."""
        self.alerts[alert.id] = alert
        if alert.status in ACTIVE_STATUSES:
            self._active_index[(alert.device_id, alert.category)] = alert.id

    def process_tick(
        self,
        device: dict,
        probability: float,
        explanation: Explanation,
        now: datetime,
    ) -> Optional[Alert]:
        """Feed one prediction in. Returns the alert that was created, updated
        or resolved, or None when nothing needed to change."""
        key = (device["device_id"], explanation.category)
        severity = severity_for(probability)

        if severity is None:
            return self._handle_low_risk(key, now)

        self._low_streak[key] = 0

        cooldown_until = self._cooldown_until.get(key)
        if cooldown_until and now < cooldown_until and severity != "CRITICAL":
            self.suppressed_total += 1
            return None

        active = self._active_alert(key)
        if active is not None:
            return self._update(active, severity, probability, explanation, now)
        return self._create(device, key, severity, probability, explanation, now)

    def _active_alert(self, key: tuple[str, str]) -> Optional[Alert]:
        alert_id = self._active_index.get(key)
        if alert_id is None:
            return None
        alert = self.alerts[alert_id]
        return alert if alert.status in ACTIVE_STATUSES else None

    def _handle_low_risk(self, key: tuple[str, str], now: datetime) -> Optional[Alert]:
        self._low_streak[key] = self._low_streak.get(key, 0) + 1
        active = self._active_alert(key)
        if active is None or self._low_streak[key] < AUTO_RESOLVE_STREAK:
            return None
        active.status = "resolved"
        active.resolved_at = now
        active.updated_at = now
        self._active_index.pop(key, None)
        return active

    def _update(
        self,
        alert: Alert,
        severity: str,
        probability: float,
        explanation: Explanation,
        now: datetime,
    ) -> Alert:
        alert.severity = severity
        alert.probability = round(probability, 4)
        alert.peak_probability = max(alert.peak_probability, probability)
        alert.eta_minutes = explanation.eta_minutes
        alert.summary = explanation.summary
        alert.root_cause = explanation.root_cause
        alert.causes = [c.__dict__ for c in explanation.causes]
        alert.recommended_actions = explanation.recommended_actions
        alert.updated_at = now
        alert.last_seen_at = now
        alert.detections_count += 1
        return alert

    def _create(
        self,
        device: dict,
        key: tuple[str, str],
        severity: str,
        probability: float,
        explanation: Explanation,
        now: datetime,
    ) -> Alert:
        alert = Alert(
            id=str(uuid.uuid4()),
            device_id=device["device_id"],
            device_name=device["name"],
            device_type=device["device_type"],
            site=device["site"],
            category=explanation.category,
            severity=severity,
            probability=round(probability, 4),
            eta_minutes=explanation.eta_minutes,
            summary=explanation.summary,
            root_cause=explanation.root_cause,
            causes=[c.__dict__ for c in explanation.causes],
            recommended_actions=explanation.recommended_actions,
            status="open",
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            peak_probability=probability,
        )
        self.alerts[alert.id] = alert
        self._active_index[key] = alert.id
        return alert

    def acknowledge(self, alert_id: str, now: datetime) -> Alert:
        alert = self.alerts[alert_id]
        if alert.status == "open":
            alert.status = "acknowledged"
            alert.updated_at = now
        return alert

    def give_feedback(self, alert_id: str, is_true_positive: bool, now: datetime) -> Alert:
        alert = self.alerts[alert_id]
        alert.feedback = "true_positive" if is_true_positive else "false_positive"
        alert.feedback_at = now
        alert.updated_at = now
        key = (alert.device_id, alert.category)

        if is_true_positive:
            # The operator confirms it is real and is working it: keep it
            # active so it stays visible until conditions actually recover.
            alert.status = "acknowledged"
        else:
            alert.status = "resolved"
            alert.resolved_at = now
            self._cooldown_until[key] = now + timedelta(minutes=COOLDOWN_MINUTES)
            if self._active_index.get(key) == alert_id:
                del self._active_index[key]
        return alert

    def list_alerts(self, status: Optional[str] = None) -> list[Alert]:
        alerts = list(self.alerts.values())
        if status:
            alerts = [a for a in alerts if a.status == status]
        return sorted(alerts, key=lambda a: a.updated_at, reverse=True)

    def active_alerts(self) -> list[Alert]:
        return [a for a in self.list_alerts() if a.status in ACTIVE_STATUSES]
