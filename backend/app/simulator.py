"""
Synthetic network telemetry generator with fault injection.

Aura-NetGuard is a research prototype. Real SNMP/Prometheus/NetFlow feeds are
not available in this environment, so this module produces statistically
realistic per-device time series (latency, loss, jitter, utilisation,
resource usage, interface errors) and injects three canonical failure modes
seen in campus/enterprise networks:

  1. gradual_degradation   - slow resource/queue build-up before an outage
                              (e.g. a router running out of buffer memory)
  2. resource_exhaustion   - CPU/memory saturation on a control-plane device
  3. link_flap             - intermittent physical-layer instability

Each fault has a lead time before the terminal "failure" event, which is
exactly what the prediction model is trained to anticipate: the goal is to
flag device state *during the lead time*, not after the outage happens.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

DEVICE_TYPES = ["core-router", "edge-switch", "access-point", "firewall", "load-balancer"]
SITES = ["Colombo-DC1", "Kandy-Branch", "Galle-Branch", "Jaffna-Branch", "HQ-Core"]

STEP_SECONDS = 30  # simulated telemetry poll interval


class FaultKind(str, Enum):
    NONE = "none"
    GRADUAL_DEGRADATION = "gradual_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LINK_FLAP = "link_flap"


class DeviceState(str, Enum):
    NORMAL = "normal"
    DEGRADING = "degrading"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class Device:
    device_id: str
    name: str
    device_type: str
    site: str


@dataclass
class MetricSnapshot:
    device_id: str
    timestamp: datetime
    latency_ms: float
    packet_loss_pct: float
    jitter_ms: float
    bandwidth_util_pct: float
    cpu_pct: float
    memory_pct: float
    interface_errors_per_min: float
    temperature_c: float
    retransmits_per_min: float
    link_up: bool
    device_state: str
    active_fault: str
    failure_event: bool  # True exactly on the step a hard failure occurs

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


def build_fleet(n_devices: int = 14, seed: Optional[int] = 7) -> list[Device]:
    rng = random.Random(seed)
    devices = []
    for i in range(n_devices):
        dtype = rng.choice(DEVICE_TYPES)
        site = rng.choice(SITES)
        devices.append(
            Device(
                device_id=f"dev-{i:03d}",
                name=f"{dtype}-{i:03d}",
                device_type=dtype,
                site=site,
            )
        )
    return devices


@dataclass
class _FaultEpisode:
    kind: FaultKind
    steps_remaining: int
    total_steps: int
    severity_target: float  # 0..1, how bad it gets at the peak


class DeviceSimulator:
    """Stateful per-device generator: call .step() once per tick."""

    BASELINES = {
        "core-router": dict(latency=2.5, loss=0.05, jitter=0.6, util=45, cpu=30, mem=40, temp=42),
        "edge-switch": dict(latency=1.2, loss=0.02, jitter=0.4, util=35, cpu=20, mem=30, temp=38),
        "access-point": dict(latency=8.0, loss=0.3, jitter=2.0, util=55, cpu=25, mem=35, temp=45),
        "firewall": dict(latency=3.0, loss=0.05, jitter=0.8, util=60, cpu=45, mem=55, temp=48),
        "load-balancer": dict(latency=1.8, loss=0.03, jitter=0.5, util=65, cpu=40, mem=50, temp=44),
    }

    FAULT_LEAD_STEPS = {
        # How many steps of build-up precede the terminal failure event.
        FaultKind.GRADUAL_DEGRADATION: (25, 45),
        FaultKind.RESOURCE_EXHAUSTION: (18, 30),
        FaultKind.LINK_FLAP: (8, 16),
    }

    def __init__(self, device: Device, seed: Optional[int] = None, fault_rate_per_1000_steps: float = 6.0):
        self.device = device
        self.rng = random.Random(seed if seed is not None else hash(device.device_id) & 0xFFFFFFFF)
        self.base = self.BASELINES[device.device_type]
        self.state = DeviceState.NORMAL
        self.episode: Optional[_FaultEpisode] = None
        self.fault_rate = fault_rate_per_1000_steps / 1000.0
        self.recovery_steps_remaining = 0
        self._t = 0

    def _maybe_start_fault(self) -> None:
        if self.episode is not None or self.state != DeviceState.NORMAL:
            return
        if self.rng.random() < self.fault_rate:
            kind = self.rng.choice(
                [FaultKind.GRADUAL_DEGRADATION, FaultKind.RESOURCE_EXHAUSTION, FaultKind.LINK_FLAP]
            )
            lo, hi = self.FAULT_LEAD_STEPS[kind]
            total = self.rng.randint(lo, hi)
            self.episode = _FaultEpisode(
                kind=kind, steps_remaining=total, total_steps=total, severity_target=self.rng.uniform(0.75, 1.0)
            )
            self.state = DeviceState.DEGRADING

    def step(self, now: datetime) -> MetricSnapshot:
        self._t += 1
        self._maybe_start_fault()

        latency = self.rng.gauss(self.base["latency"], self.base["latency"] * 0.08)
        loss = max(0.0, self.rng.gauss(self.base["loss"], self.base["loss"] * 0.5 + 0.01))
        jitter = max(0.0, self.rng.gauss(self.base["jitter"], self.base["jitter"] * 0.1))
        util = max(0.0, min(100.0, self.rng.gauss(self.base["util"], 4)))
        cpu = max(0.0, min(100.0, self.rng.gauss(self.base["cpu"], 3)))
        mem = max(0.0, min(100.0, self.rng.gauss(self.base["mem"], 3)))
        temp = max(0.0, self.rng.gauss(self.base["temp"], 1.2))
        iface_err = max(0.0, self.rng.gauss(0.1, 0.1))
        retrans = max(0.0, self.rng.gauss(0.2, 0.2))
        link_up = True
        failure_event = False
        active_fault = FaultKind.NONE

        if self.episode is not None:
            active_fault = self.episode.kind
            progress = 1.0 - (self.episode.steps_remaining / self.episode.total_steps)
            ramp = self._ease(progress) * self.episode.severity_target

            if self.episode.kind == FaultKind.GRADUAL_DEGRADATION:
                latency += ramp * self.base["latency"] * 12
                loss += ramp * 25
                jitter += ramp * self.base["jitter"] * 8
                util = min(100.0, util + ramp * 40)
                cpu = min(100.0, cpu + ramp * 30)
            elif self.episode.kind == FaultKind.RESOURCE_EXHAUSTION:
                cpu = min(100.0, cpu + ramp * 65)
                mem = min(100.0, mem + ramp * 55)
                temp += ramp * 25
                latency += ramp * self.base["latency"] * 6
                iface_err += ramp * 3
            elif self.episode.kind == FaultKind.LINK_FLAP:
                if self.rng.random() < 0.3 + 0.5 * ramp:
                    loss += self.rng.uniform(10, 40) * ramp
                    retrans += ramp * 8
                    iface_err += ramp * 5
                    if self.rng.random() < 0.15 * ramp:
                        link_up = False

            self.episode.steps_remaining -= 1
            if self.episode.steps_remaining <= 0:
                failure_event = True
                self.state = DeviceState.FAILED
                link_up = False if self.episode.kind != FaultKind.GRADUAL_DEGRADATION else link_up
                loss = max(loss, 60.0)
                self.recovery_steps_remaining = self.rng.randint(4, 10)
                self.episode = None
            elif self.episode.steps_remaining <= max(3, self.episode.total_steps // 4):
                self.state = DeviceState.CRITICAL
            else:
                self.state = DeviceState.DEGRADING

        elif self.state == DeviceState.FAILED:
            self.state = DeviceState.RECOVERING
            link_up = False
            loss = max(loss, 50.0)

        elif self.state == DeviceState.RECOVERING:
            self.recovery_steps_remaining -= 1
            recovery_progress = 1 - max(0, self.recovery_steps_remaining) / 10
            loss = max(0.0, loss * (1 - recovery_progress) + self.base["loss"] * recovery_progress)
            link_up = self.recovery_steps_remaining <= 6
            if self.recovery_steps_remaining <= 0:
                self.state = DeviceState.NORMAL

        return MetricSnapshot(
            device_id=self.device.device_id,
            timestamp=now,
            latency_ms=round(max(0.0, latency), 3),
            packet_loss_pct=round(min(100.0, max(0.0, loss)), 3),
            jitter_ms=round(max(0.0, jitter), 3),
            bandwidth_util_pct=round(util, 2),
            cpu_pct=round(cpu, 2),
            memory_pct=round(mem, 2),
            interface_errors_per_min=round(max(0.0, iface_err), 3),
            temperature_c=round(temp, 2),
            retransmits_per_min=round(max(0.0, retrans), 3),
            link_up=link_up,
            device_state=self.state.value,
            active_fault=active_fault.value if isinstance(active_fault, FaultKind) else active_fault,
            failure_event=failure_event,
        )

    @staticmethod
    def _ease(progress: float) -> float:
        # smoothstep-ish acceleration curve so degradation is subtle at first,
        # then compounds sharply near the failure point (matches real
        # queueing/thermal/backoff dynamics better than a linear ramp).
        progress = min(1.0, max(0.0, progress))
        return progress * progress * (3 - 2 * progress)


class FleetSimulator:
    """Drives many DeviceSimulators together and advances a shared clock."""

    def __init__(self, devices: list[Device], start_time: Optional[datetime] = None, seed: Optional[int] = 7):
        rng = random.Random(seed)
        self.devices = devices
        self.sims = {d.device_id: DeviceSimulator(d, seed=rng.randint(0, 1_000_000)) for d in devices}
        self.clock = start_time or datetime.utcnow()

    def step(self) -> list[MetricSnapshot]:
        self.clock += timedelta(seconds=STEP_SECONDS)
        return [sim.step(self.clock) for sim in self.sims.values()]

    def run(self, n_steps: int) -> list[MetricSnapshot]:
        out: list[MetricSnapshot] = []
        for _ in range(n_steps):
            out.extend(self.step())
        return out
