from datetime import datetime, timedelta

from backend.app.alerts.engine import AUTO_RESOLVE_STREAK, AlertEngine
from backend.app.alerts.explain import CauseDetail, Explanation
from backend.app.alerts.models import severity_for

DEVICE = {
    "device_id": "dev-000",
    "name": "core-router-000",
    "device_type": "core-router",
    "site": "HQ-Core",
}


def _explanation(category="resource_exhaustion"):
    return Explanation(
        category=category,
        root_cause="test root cause",
        summary="test summary",
        causes=[CauseDetail("cpu_pct", "CPU utilisation", 91.0, "%", 0.2)],
        recommended_actions=["do the thing"],
        eta_minutes=5.0,
    )


def test_severity_thresholds():
    assert severity_for(0.9) == "CRITICAL"
    assert severity_for(0.7) == "HIGH"
    assert severity_for(0.5) == "MEDIUM"
    assert severity_for(0.3) == "LOW"
    assert severity_for(0.1) is None


def test_repeated_ticks_update_one_alert_instead_of_creating_new_ones():
    engine = AlertEngine()
    now = datetime.utcnow()
    first = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    second = engine.process_tick(DEVICE, 0.92, _explanation(), now + timedelta(seconds=30))
    assert first.id == second.id
    assert second.detections_count == 2
    assert len(engine.list_alerts()) == 1


def test_peak_probability_survives_a_drop():
    engine = AlertEngine()
    now = datetime.utcnow()
    engine.process_tick(DEVICE, 0.92, _explanation(), now)
    alert = engine.process_tick(DEVICE, 0.5, _explanation(), now + timedelta(seconds=30))
    assert alert.severity == "MEDIUM"
    assert alert.peak_probability == 0.92


def test_different_category_on_same_device_creates_a_separate_alert():
    engine = AlertEngine()
    now = datetime.utcnow()
    a = engine.process_tick(DEVICE, 0.9, _explanation("resource_exhaustion"), now)
    b = engine.process_tick(DEVICE, 0.9, _explanation("thermal"), now)
    assert a.id != b.id
    assert len(engine.list_alerts()) == 2


def test_low_probability_raises_no_alert():
    engine = AlertEngine()
    assert engine.process_tick(DEVICE, 0.1, _explanation(), datetime.utcnow()) is None
    assert engine.list_alerts() == []


def test_alert_auto_resolves_after_sustained_low_risk():
    engine = AlertEngine()
    now = datetime.utcnow()
    assert engine.process_tick(DEVICE, 0.9, _explanation(), now).status == "open"

    result = None
    for _ in range(AUTO_RESOLVE_STREAK):
        now += timedelta(seconds=30)
        result = engine.process_tick(DEVICE, 0.05, _explanation(), now)

    assert result.status == "resolved"
    assert result.resolved_at is not None


def test_auto_resolve_needs_a_sustained_streak_not_one_quiet_tick():
    engine = AlertEngine()
    now = datetime.utcnow()
    engine.process_tick(DEVICE, 0.9, _explanation(), now)
    assert engine.process_tick(DEVICE, 0.05, _explanation(), now + timedelta(seconds=30)) is None
    assert engine.active_alerts()[0].status == "open"


def test_false_positive_feedback_suppresses_the_same_device_and_cause():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    engine.give_feedback(alert.id, is_true_positive=False, now=now)

    assert engine.process_tick(DEVICE, 0.7, _explanation(), now + timedelta(minutes=1)) is None
    assert engine.suppressed_total == 1


def test_cooldown_does_not_leak_to_a_different_cause():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation("resource_exhaustion"), now)
    engine.give_feedback(alert.id, is_true_positive=False, now=now)

    other = engine.process_tick(DEVICE, 0.7, _explanation("thermal"), now + timedelta(minutes=1))
    assert other is not None
    assert other.category == "thermal"


def test_critical_risk_breaks_through_the_cooldown():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    engine.give_feedback(alert.id, is_true_positive=False, now=now)

    escalated = engine.process_tick(DEVICE, 0.95, _explanation(), now + timedelta(minutes=1))
    assert escalated is not None
    assert escalated.severity == "CRITICAL"


def test_cooldown_expires():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    engine.give_feedback(alert.id, is_true_positive=False, now=now)

    later = engine.process_tick(DEVICE, 0.7, _explanation(), now + timedelta(minutes=25))
    assert later is not None
    assert later.status == "open"


def test_true_positive_feedback_keeps_the_alert_active():
    engine = AlertEngine()
    now = datetime.utcnow()
    alert = engine.process_tick(DEVICE, 0.9, _explanation(), now)
    updated = engine.give_feedback(alert.id, is_true_positive=True, now=now)
    assert updated.status == "acknowledged"
    assert updated.feedback == "true_positive"
    assert updated in engine.active_alerts()


def test_loaded_alerts_are_deduplicated_against_not_duplicated():
    """An alert restored from storage must still be the one that gets updated."""
    engine = AlertEngine()
    now = datetime.utcnow()
    original = engine.process_tick(DEVICE, 0.9, _explanation(), now)

    restored = AlertEngine()
    restored.load_alert(original)
    updated = restored.process_tick(DEVICE, 0.91, _explanation(), now + timedelta(seconds=30))
    assert updated.id == original.id
    assert len(restored.list_alerts()) == 1
