from datetime import datetime, timedelta

import pytest

from backend.app.collectors.counters import (
    COUNTER32_MAX,
    CounterTracker,
    per_minute,
    utilisation_pct,
)

KEY = ("dev-001", "ifInErrors", "Gi0/1")
T0 = datetime(2026, 1, 1, 12, 0, 0)


def test_first_reading_only_sets_a_baseline():
    tracker = CounterTracker()
    assert tracker.update(KEY, 1000, T0) is None


def test_steady_increase_gives_the_expected_rate():
    tracker = CounterTracker()
    tracker.update(KEY, 1000, T0)
    rate = tracker.update(KEY, 1300, T0 + timedelta(seconds=30))
    assert rate == pytest.approx(10.0)  # 300 errors over 30s


def test_no_change_is_a_zero_rate_not_none():
    """A quiet counter is real information: the rate is zero."""
    tracker = CounterTracker()
    tracker.update(KEY, 1000, T0)
    assert tracker.update(KEY, 1000, T0 + timedelta(seconds=30)) == 0.0


def test_counter32_wrap_is_accounted_for_not_spiked():
    tracker = CounterTracker()
    tracker.update(KEY, COUNTER32_MAX - 100, T0)
    rate = tracker.update(KEY, 50, T0 + timedelta(seconds=30))
    # 100 to reach the ceiling + 50 after wrapping = 150 over 30s
    assert rate == pytest.approx(5.0)


def test_reboot_detected_by_uptime_returns_no_rate():
    """The authoritative reset signal: sysUpTime went backwards."""
    tracker = CounterTracker()
    tracker.update(KEY, 5_000_000, T0, uptime_ticks=900_000)
    rate = tracker.update(KEY, 120, T0 + timedelta(seconds=30), uptime_ticks=300)
    assert rate is None


def test_after_a_reboot_the_next_interval_measures_again():
    tracker = CounterTracker()
    tracker.update(KEY, 5_000_000, T0, uptime_ticks=900_000)
    tracker.update(KEY, 120, T0 + timedelta(seconds=30), uptime_ticks=300)
    rate = tracker.update(KEY, 300, T0 + timedelta(seconds=60), uptime_ticks=3300)
    assert rate == pytest.approx(6.0)  # 180 errors over 30s, counted from the new baseline


def test_reset_without_uptime_is_caught_by_rate_plausibility():
    """Without sysUpTime, a reset still must not become a fake spike: reading
    it as a wrap would imply billions of errors per second."""
    tracker = CounterTracker(width=32, max_plausible_rate=1_000_000)
    tracker.update(KEY, 5_000_000, T0)
    assert tracker.update(KEY, 120, T0 + timedelta(seconds=30)) is None


def test_a_genuine_wrap_near_the_ceiling_is_still_allowed():
    """The plausibility guard must not reject real wraps."""
    tracker = CounterTracker(width=32, max_plausible_rate=1_000_000)
    tracker.update(KEY, COUNTER32_MAX - 1000, T0)
    rate = tracker.update(KEY, 500, T0 + timedelta(seconds=30))
    assert rate == pytest.approx(50.0)


def test_implausible_forward_jump_is_rejected():
    tracker = CounterTracker(max_plausible_rate=1000)
    tracker.update(KEY, 100, T0)
    assert tracker.update(KEY, 10_000_000, T0 + timedelta(seconds=1)) is None


def test_zero_or_negative_elapsed_time_is_rejected():
    tracker = CounterTracker()
    tracker.update(KEY, 100, T0)
    assert tracker.update(KEY, 200, T0) is None
    assert tracker.update(KEY, 300, T0 - timedelta(seconds=5)) is None


def test_keys_are_tracked_independently():
    tracker = CounterTracker()
    other = ("dev-001", "ifInErrors", "Gi0/2")
    tracker.update(KEY, 1000, T0)
    tracker.update(other, 50, T0)
    assert tracker.update(KEY, 1060, T0 + timedelta(seconds=60)) == pytest.approx(1.0)
    assert tracker.update(other, 110, T0 + timedelta(seconds=60)) == pytest.approx(1.0)


def test_counter64_does_not_wrap_at_the_32bit_ceiling():
    tracker = CounterTracker(width=64)
    tracker.update(KEY, COUNTER32_MAX + 1000, T0)
    rate = tracker.update(KEY, COUNTER32_MAX + 1600, T0 + timedelta(seconds=60))
    assert rate == pytest.approx(10.0)


def test_explicit_reset_clears_the_baseline():
    tracker = CounterTracker()
    tracker.update(KEY, 1000, T0)
    tracker.reset(KEY)
    assert tracker.update(KEY, 1200, T0 + timedelta(seconds=30)) is None


def test_invalid_width_is_rejected():
    with pytest.raises(ValueError):
        CounterTracker(width=16)


def test_utilisation_converts_octets_to_link_percentage():
    # 12.5 MB/s = 100 Mbit/s on a 1 Gbit link = 10%
    assert utilisation_pct(12_500_000, 1_000_000_000) == pytest.approx(10.0)


def test_utilisation_is_clamped_and_safe_on_unknown_speed():
    assert utilisation_pct(10**12, 1_000_000_000) == 100.0
    assert utilisation_pct(1000, 0) == 0.0


def test_per_minute_passes_none_through():
    assert per_minute(None) is None
    assert per_minute(2.0) == pytest.approx(120.0)
