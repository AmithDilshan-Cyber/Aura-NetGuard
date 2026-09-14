"""In-process runtime state: the live fleet simulator, rolling metric
history per device, and the alert engine. A single instance of this lives
on `app.state.runtime` for the lifetime of the FastAPI process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .alerts.engine import AlertEngine
from .features import WINDOW_STEPS
from .simulator import Device, FleetSimulator, build_fleet

CHART_HISTORY_LEN = 240  # 2 hours of 30s ticks kept in memory for the dashboard


@dataclass
class RuntimeState:
    fleet: FleetSimulator
    devices_by_id: dict[str, Device]
    history: dict[str, deque] = field(default_factory=dict)
    latest: dict[str, dict] = field(default_factory=dict)
    engine: AlertEngine = field(default_factory=AlertEngine)
    ticks: int = 0
    model_available: bool = False

    @classmethod
    def create(cls, n_devices: int = 14, seed: int = 7) -> "RuntimeState":
        devices = build_fleet(n_devices=n_devices, seed=seed)
        fleet = FleetSimulator(devices, seed=seed)
        state = cls(fleet=fleet, devices_by_id={d.device_id: d for d in devices})
        for d in devices:
            state.history[d.device_id] = deque(maxlen=CHART_HISTORY_LEN)
        return state

    def feature_window(self, device_id: str) -> list[dict] | None:
        hist = self.history[device_id]
        if len(hist) < WINDOW_STEPS:
            return None
        return list(hist)[-WINDOW_STEPS:]
