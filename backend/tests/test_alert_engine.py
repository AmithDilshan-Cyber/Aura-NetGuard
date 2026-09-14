from datetime import datetime, timedelta

from backend.app.alerts.engine import AUTO_RESOLVE_STREAK, COOLDOWN_MINUTES, AlertEngine
from backend.app.alerts.explain import Explanation, CauseDetail
from backend.app.alerts.models import severity_for

DEVICE = {"device_id": "dev-000", "name": "core-router-000", "device_type": "core-router", "site": "HQ-Core"}


def _explanation(category="resource_exhaustion"):
    return Explanation(
        category=category,
        root_cause="test root cause",
        summary="test summary",
        causes=[CauseDetail(metric="cpu_pct", label="CPU utilisation", value=91.0, unit="%", contribution=0.2)],
        recommended_actions=["do the thing"],
        eta_minutes=5.0,
    )


def test_severity_thresholds():
    assert severity_for(0.9) == "CRITICAL"
    assert severity_for(0.7) == "HIGH"
    assert severity_for(0.5) == "MEDIUM"
    assert severity_for(0.3) == "LOW"
    assert severity_for(0.1) is None


def test_repeated_ticks_update_same_alert_not_create_new():
    engine = AlertEngine()
    now = datetime.utcnow()
    a1 = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    a2 = engine.process_tick(DEVICE, 0.92, _explanation(), now + timedelta(seconds=30))
    assert a1.id == a2.id
    assert a2.detections_count == 2
    assert len(engine.list_alerts()) == 1


def test_different_category_creates_separate_alert():
    engine = AlertEngine()
    now = datetime.utcnow()
    a1 = engine.process_tick(DEVICE, 0.9, _explanation("resource_exhaustion"), now)
    a2 = engine.process_tick(DEVICE, 0.9, _explanation("thermal"), now)
    assert a1.id != a2.id
    assert len(engine.list_alerts()) == 2


def test_low_probability_does_not_create_alert():
    engine = AlertEngine()
    now = datetime.utcnow()
    result = engine.process_tick(DEVICE, 0.1, _explanation(), now)
    assert result is None
    assert len(engine.list_alerts()) == 0


def test_auto_resolve_after_sustained_low_risk():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    assert alert.status == "open"

    result = None
    for i in range(AUTO_RESOLVE_STREAK):
        now += timedelta(seconds=30)
        result = engine.process_tick(DEVICE, 0.05, _explanation(), now)

    assert result is not None
    assert result.status == "resolved"


def test_false_positive_feedback_triggers_cooldown_suppression():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    engine.give_feedback(alert.id, is_true_positive=False, now=now)

    # same device+category, still within cooldown, same (non-critical) severity -> suppressed
    result = engine.process_tick(DEVICE, 0.7, _explanation(), now + timedelta(minutes=1))
    assert result is None
    assert engine.suppressed_total == 1


def test_critical_severity_breaks_through_cooldown():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    engine.give_feedback(alert.id, is_true_positive=False, now=now)

    result = engine.process_tick(DEVICE, 0.95, _explanation(), now + timedelta(minutes=1))
    assert result is not None
    assert result.severity == "CRITICAL"


def test_true_positive_feedback_keeps_alert_active_as_acknowledged():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    updated = engine.give_feedback(alert.id, is_true_positive=True, now=now)
    assert updated.status == "acknowledged"
    assert updated.feedback == "true_positive"
