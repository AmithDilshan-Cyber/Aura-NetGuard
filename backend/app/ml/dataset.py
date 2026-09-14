"""Builds a labelled training set from the synthetic simulator.

Label definition: a window is positive (1) if the device experiences a
`failure_event` at any point within the next PREDICTION_HORIZON_STEPS steps.
This is what makes the model a *predictor* rather than an anomaly detector:
it is trained to recognise the lead-up signature, not the failure itself.
Steps where the device is already in the "failed" state are excluded --
predicting a failure that has already happened is not a useful task.
"""

from __future__ import annotations

import numpy as np

from ..features import (
    PREDICTION_HORIZON_STEPS,
    WINDOW_STEPS,
    build_model_vector,
    extract_features,
    model_feature_names,
)
from ..simulator import Device, DeviceSimulator, build_fleet
from datetime import datetime, timedelta

TRAIN_METRIC_KEYS = [
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
    "bandwidth_util_pct",
    "cpu_pct",
    "memory_pct",
    "interface_errors_per_min",
    "temperature_c",
    "retransmits_per_min",
]


def _simulate_device_history(device: Device, n_steps: int, seed: int, fault_rate: float) -> list[dict]:
    sim = DeviceSimulator(device, seed=seed, fault_rate_per_1000_steps=fault_rate)
    now = datetime.utcnow()
    history = []
    for i in range(n_steps):
        snap = sim.step(now + timedelta(seconds=30 * i))
        history.append(snap.to_dict())
    return history


def build_dataset(
    n_devices: int = 20,
    steps_per_device: int = 4000,
    fault_rate_per_1000_steps: float = 18.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, device_group_ids, feature_names).

    device_group_ids lets callers do a device-level train/test split so the
    reported accuracy reflects generalisation to *unseen devices*, not just
    unseen timestamps on devices the model already saw (a common source of
    inflated results in network telemetry ML work).
    """
    devices = build_fleet(n_devices=n_devices, seed=seed)
    rng_seed = seed

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    group_rows: list[int] = []

    for di, device in enumerate(devices):
        history = _simulate_device_history(
            device, steps_per_device, seed=rng_seed + di, fault_rate=fault_rate_per_1000_steps
        )
        n = len(history)
        for t in range(WINDOW_STEPS - 1, n - PREDICTION_HORIZON_STEPS):
            if history[t]["device_state"] == "failed":
                continue
            window = history[t - WINDOW_STEPS + 1 : t + 1]
            feats = extract_features(window)
            vec = build_model_vector(feats, device.device_type)

            future = history[t + 1 : t + 1 + PREDICTION_HORIZON_STEPS]
            label = 1 if any(row["failure_event"] for row in future) else 0

            X_rows.append(vec)
            y_rows.append(label)
            group_rows.append(di)

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=int)
    groups = np.array(group_rows, dtype=int)
    return X, y, groups, model_feature_names()
