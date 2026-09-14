from datetime import datetime

from backend.app.simulator import DeviceSimulator, FleetSimulator, build_fleet


def test_build_fleet_is_deterministic_with_seed():
    a = build_fleet(n_devices=10, seed=5)
    b = build_fleet(n_devices=10, seed=5)
    assert [d.device_id for d in a] == [d.device_id for d in b]
    assert [d.device_type for d in a] == [d.device_type for d in b]


def test_device_simulator_produces_valid_ranges():
    devices = build_fleet(n_devices=1, seed=1)
    sim = DeviceSimulator(devices[0], seed=1, fault_rate_per_1000_steps=50)
    now = datetime.utcnow()
    for i in range(500):
        snap = sim.step(now)
        assert 0.0 <= snap.packet_loss_pct <= 100.0
        assert 0.0 <= snap.cpu_pct <= 100.0
        assert 0.0 <= snap.memory_pct <= 100.0
        assert snap.latency_ms >= 0.0
        assert snap.device_state in {"normal", "degrading", "critical", "failed", "recovering"}


def test_fault_eventually_produces_a_failure_event():
    devices = build_fleet(n_devices=1, seed=2)
    sim = DeviceSimulator(devices[0], seed=2, fault_rate_per_1000_steps=400)
    now = datetime.utcnow()
    failures = 0
    for i in range(1000):
        snap = sim.step(now)
        if snap.failure_event:
            failures += 1
    assert failures > 0


def test_fleet_simulator_advances_clock_and_returns_all_devices():
    devices = build_fleet(n_devices=6, seed=3)
    fleet = FleetSimulator(devices, seed=3)
    t0 = fleet.clock
    snapshots = fleet.step()
    assert len(snapshots) == 6
    assert fleet.clock > t0
    device_ids = {s.device_id for s in snapshots}
    assert device_ids == {d.device_id for d in devices}
