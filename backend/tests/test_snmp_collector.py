"""Collector pipeline tests against a fake SNMP transport.

The pysnmp transport itself cannot be exercised without hardware, which is
exactly why it is isolated behind `SnmpSession`: everything above it -- rate
conversion, aggregation, snapshot assembly, failure handling -- is tested
here.
"""

from datetime import datetime, timedelta

import pytest

from backend.app.collectors.icmp import ProbeResult
from backend.app.collectors.inventory import DeviceTarget
from backend.app.collectors.snmp import (
    UNREACHABLE_LATENCY_MS,
    SNMPCollector,
)

T0 = datetime(2026, 1, 1, 12, 0, 0)

TARGET = DeviceTarget(
    device_id="dev-001",
    name="core-rtr",
    device_type="core-router",
    site="Main",
    host="10.0.0.1",
    interfaces=[1],
)


class FakeSession:
    """Serves scripted SNMP values; advance with `tick()`."""

    def __init__(self, scalars, interfaces):
        self.scalars = scalars
        self.interfaces = interfaces
        self.index = 0

    def tick(self):
        self.index += 1

    def get(self, target, oids):
        return dict(self.scalars[min(self.index, len(self.scalars) - 1)])

    def get_interface(self, target, if_index, oids):
        return dict(self.interfaces[min(self.index, len(self.interfaces) - 1)])


def _scalars(uptime, cpu=30.0, used=400.0, free=600.0, temp=42.0, retrans=100):
    return {
        "sysUpTime": uptime,
        "cpu_pct": cpu,
        "memory_used": used,
        "memory_free": free,
        "temperature_c": temp,
        "tcpRetransSegs": retrans,
    }


def _interface(errors=0, octets=0, speed=1000):
    return {
        "ifInErrors": errors,
        "ifOutErrors": 0,
        "ifInDiscards": 0,
        "ifHCInOctets": octets,
        "ifHCOutOctets": 0,
        "ifHighSpeed": speed,  # Mbit/s
    }


def _collector(session, probe_result=ProbeResult(0.0, 12.5, 0.8), clock_steps=None):
    steps = iter(clock_steps or [T0, T0 + timedelta(seconds=30)])
    return SNMPCollector(
        [TARGET],
        session=session,
        probe=lambda host: probe_result,
        clock=lambda: next(steps),
    )


def test_first_poll_yields_nothing_because_counters_need_a_baseline():
    session = FakeSession([_scalars(1000)], [_interface()])
    assert _collector(session).step() == []


def test_second_poll_produces_a_snapshot_with_real_rates():
    session = FakeSession(
        [_scalars(1000, retrans=100), _scalars(4000, retrans=160)],
        [_interface(errors=0, octets=0), _interface(errors=60, octets=37_500_000)],
    )
    collector = _collector(session)

    collector.step()
    session.tick()
    snapshots = collector.step()

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.device_id == "dev-001"
    assert snap.interface_errors_per_min == pytest.approx(120.0)  # 60 errors / 30s
    assert snap.retransmits_per_min == pytest.approx(120.0)       # 60 segs / 30s
    # 37.5 MB over 30s = 1.25 MB/s = 10 Mbit/s on a 1000 Mbit link = 1%
    assert snap.bandwidth_util_pct == pytest.approx(1.0)
    assert snap.memory_pct == pytest.approx(40.0)
    assert snap.cpu_pct == pytest.approx(30.0)


def test_probe_values_populate_latency_jitter_and_loss():
    session = FakeSession([_scalars(1000), _scalars(4000)], [_interface(), _interface()])
    collector = _collector(session, probe_result=ProbeResult(25.0, 88.4, 9.1))

    collector.step()
    session.tick()
    snap = collector.step()[0]

    assert snap.packet_loss_pct == pytest.approx(25.0)
    assert snap.latency_ms == pytest.approx(88.4)
    assert snap.jitter_ms == pytest.approx(9.1)
    assert snap.link_up is True


def test_device_down_since_startup_still_reaches_the_dashboard():
    """Total loss is the most important signal this system exists to show.
    It must survive even with no RTT and no prior reading to fall back on."""
    session = FakeSession([_scalars(1000), _scalars(4000)], [_interface(), _interface()])
    collector = _collector(session, probe_result=ProbeResult(100.0, None, None))

    collector.step()
    session.tick()
    snapshots = collector.step()

    assert len(snapshots) == 1, "an unreachable device must not be silently dropped"
    snap = snapshots[0]
    assert snap.packet_loss_pct == 100.0
    assert snap.link_up is False
    assert snap.device_state == "failed"
    assert snap.latency_ms == pytest.approx(UNREACHABLE_LATENCY_MS)


def test_device_that_goes_down_keeps_its_last_known_latency():
    """Once a device has answered, its real last reading is a better
    stand-in than the generic sentinel."""
    session = FakeSession(
        [_scalars(1000), _scalars(4000), _scalars(7000)],
        [_interface(), _interface(errors=10), _interface(errors=20)],
    )
    probes = iter([
        ProbeResult(0.0, 33.3, 2.2),    # healthy
        ProbeResult(0.0, 33.3, 2.2),
        ProbeResult(100.0, None, None),  # goes down
    ])
    steps = iter([T0, T0 + timedelta(seconds=30), T0 + timedelta(seconds=60)])
    collector = SNMPCollector(
        [TARGET], session=session,
        probe=lambda host: next(probes),
        clock=lambda: next(steps),
    )

    collector.step()
    session.tick()
    collector.step()
    session.tick()
    snap = collector.step()[0]

    assert snap.packet_loss_pct == 100.0
    assert snap.latency_ms == pytest.approx(33.3)
    assert snap.jitter_ms == pytest.approx(2.2)


def test_device_reboot_suppresses_the_interval_rather_than_spiking():
    """Uptime goes backwards and counters reset: no fabricated error spike."""
    session = FakeSession(
        [_scalars(900_000, retrans=5_000_000), _scalars(300, retrans=20)],
        [_interface(errors=9_000_000), _interface(errors=5)],
    )
    collector = _collector(session)

    collector.step()
    session.tick()
    assert collector.step() == []  # no trustworthy rate -> no snapshot


def test_device_without_vendor_oids_is_still_collected():
    """CPU/memory/temperature OIDs are vendor-specific and missing on lots of
    hardware. Requiring them would mean collecting nothing from those
    devices, when their interface counters are still valuable."""
    bare = {k: None for k in _scalars(4000)}
    bare["sysUpTime"] = 4000
    first = {k: None for k in _scalars(1000)}
    first["sysUpTime"] = 1000

    session = FakeSession([first, bare], [_interface(), _interface(errors=60)])
    collector = _collector(session)

    collector.step()
    session.tick()
    snapshots = collector.step()

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.interface_errors_per_min == pytest.approx(120.0)  # the real signal survives
    assert snap.cpu_pct == 0.0       # unavailable -> constant, carries no trend
    assert snap.temperature_c == 0.0


def test_missing_required_metric_skips_the_device_rather_than_guessing():
    """Interface counters are the point; without them there is nothing to say."""
    session = FakeSession(
        [_scalars(1000), _scalars(4000)],
        [_interface(), {**_interface(errors=10), "ifHCInOctets": None,
                        "ifHCOutOctets": None, "ifHighSpeed": None}],
    )
    collector = _collector(session)

    collector.step()
    session.tick()
    assert collector.step() == []


def test_a_failing_device_does_not_stop_the_rest_of_the_fleet():
    class ExplodingSession(FakeSession):
        def get(self, target, oids):
            if target.device_id == "dev-bad":
                raise TimeoutError("device unreachable over SNMP")
            return super().get(target, oids)

    bad = DeviceTarget(
        device_id="dev-bad", name="bad", device_type="edge-switch",
        site="Main", host="10.0.0.9", interfaces=[1],
    )
    session = ExplodingSession(
        [_scalars(1000), _scalars(4000)], [_interface(), _interface(errors=10)]
    )
    steps = iter([T0, T0 + timedelta(seconds=30)])
    collector = SNMPCollector(
        [bad, TARGET],
        session=session,
        probe=lambda host: ProbeResult(0.0, 12.5, 0.8),
        clock=lambda: next(steps),
    )

    collector.step()
    session.tick()
    snapshots = collector.step()

    assert [s.device_id for s in snapshots] == ["dev-001"]


def test_real_collectors_cannot_claim_ground_truth_labels():
    """failure_event is simulator-only; real data needs external labels."""
    session = FakeSession([_scalars(1000), _scalars(4000)], [_interface(), _interface()])
    collector = _collector(session)

    collector.step()
    session.tick()
    snap = collector.step()[0]

    assert snap.failure_event is False
    assert snap.active_fault == "unknown"


def test_collector_exposes_devices_for_the_dashboard():
    session = FakeSession([_scalars(1000)], [_interface()])
    collector = _collector(session)
    assert [d.device_id for d in collector.devices] == ["dev-001"]
    assert collector.devices[0].device_type == "core-router"
