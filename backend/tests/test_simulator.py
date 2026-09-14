from datetime import datetime

from backend.app.simulator import DeviceSimulator, FleetSimulator, build_fleet


def test_build_fleet_is_deterministic_with_seed():
    a = build_fleet(n_devices=10, seed=5)
    b = build_fleet(n_devices=10, seed=5)
    assert [d.device_id for d in a] == [d.device_id for d in b]
    assert [d.device_type for d in a] == [d.device_type for d in b]


def test_metrics_stay_in_valid_ranges_even_under_faults():
    device = build_fleet(n_devices=1, seed=1)[0]
    sim = DeviceSimulator(device, seed=1, fault_rate_per_1000_steps=50)
    now = datetime.utcnow()
    for _ in range(500):
        snap = sim.step(now)
        assert 0.0 <= snap.packet_loss_pct <= 100.0
        assert 0.0 <= snap.cpu_pct <= 100.0
        assert 0.0 <= snap.memory_pct <= 100.0
        assert snap.latency_ms >= 0.0
        assert snap.device_state in {"normal", "degrading", "critical", "failed", "recovering"}


def test_faults_eventually_produce_failure_events():
    device = build_fleet(n_devices=1, seed=2)[0]
    sim = DeviceSimulator(device, seed=2, fault_rate_per_1000_steps=400)
    now = datetime.utcnow()
    failures = sum(1 for _ in range(1000) if sim.step(now).failure_event)
    assert failures > 0


def test_degradation_precedes_the_failure_event():
    """The whole premise of the model is that a failure has a lead-in period."""
    device = build_fleet(n_devices=1, seed=4)[0]
    sim = DeviceSimulator(device, seed=4, fault_rate_per_1000_steps=400)
    now = datetime.utcnow()

    states = []
    for _ in range(1000):
        snap = sim.step(now)
        states.append(snap.device_state)
        if snap.failure_event:
            break

    assert states[-1] == "failed"
    assert "degrading" in states[:-1]


def test_no_fault_injection_stays_normal():
    device = build_fleet(n_devices=1, seed=6)[0]
    sim = DeviceSimulator(device, seed=6, fault_rate_per_1000_steps=0.0)
    now = datetime.utcnow()
    for _ in range(300):
        snap = sim.step(now)
        assert snap.device_state == "normal"
        assert snap.active_fault == "none"
        assert not snap.failure_event


def test_fleet_simulator_advances_clock_and_returns_all_devices():
    devices = build_fleet(n_devices=6, seed=3)
    fleet = FleetSimulator(devices, seed=3)
    t0 = fleet.clock
    snapshots = fleet.step()
    assert len(snapshots) == 6
    assert fleet.clock > t0
    assert {s.device_id for s in snapshots} == {d.device_id for d in devices}
