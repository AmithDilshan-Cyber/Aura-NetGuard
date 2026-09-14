"""Rolling-window feature engineering shared by training and live inference.

Using the *same* function for both paths is deliberate: a common source of
silent train/serve skew in failure-prediction systems is featurizing
historical data one way and live data another. Here there is exactly one
implementation of "what a feature vector looks like".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .simulator import DEVICE_TYPES

METRICS = [
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

WINDOW_STEPS = 10          # 10 * 30s = 5 minutes of history per feature vector
PREDICTION_HORIZON_STEPS = 15  # look 15 steps (~7.5 min) ahead when labelling

# Human-readable metric labels + the direction that indicates trouble, used
# by the alert explainer to turn a feature name into a sentence.
METRIC_META = {
    "latency_ms": {"label": "round-trip latency", "unit": "ms", "bad_direction": "up"},
    "packet_loss_pct": {"label": "packet loss", "unit": "%", "bad_direction": "up"},
    "jitter_ms": {"label": "jitter", "unit": "ms", "bad_direction": "up"},
    "bandwidth_util_pct": {"label": "bandwidth utilisation", "unit": "%", "bad_direction": "up"},
    "cpu_pct": {"label": "CPU utilisation", "unit": "%", "bad_direction": "up"},
    "memory_pct": {"label": "memory utilisation", "unit": "%", "bad_direction": "up"},
    "interface_errors_per_min": {"label": "interface errors", "unit": "/min", "bad_direction": "up"},
    "temperature_c": {"label": "chassis temperature", "unit": "°C", "bad_direction": "up"},
    "retransmits_per_min": {"label": "TCP retransmits", "unit": "/min", "bad_direction": "up"},
}


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    # least-squares slope; cheap and robust enough for a 10-point window
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def feature_names() -> list[str]:
    names = []
    for metric in METRICS:
        for stat in ("last", "mean", "std", "slope", "delta", "max"):
            names.append(f"{metric}__{stat}")
    return names


def extract_features(window: Sequence[dict]) -> dict[str, float]:
    """window: list of metric-snapshot dicts, oldest -> newest, length <= WINDOW_STEPS."""
    feats: dict[str, float] = {}
    for metric in METRICS:
        values = np.array([row[metric] for row in window], dtype=float)
        feats[f"{metric}__last"] = float(values[-1])
        feats[f"{metric}__mean"] = float(values.mean())
        feats[f"{metric}__std"] = float(values.std())
        feats[f"{metric}__slope"] = _slope(values)
        feats[f"{metric}__delta"] = float(values[-1] - values[0])
        feats[f"{metric}__max"] = float(values.max())
    return feats


def vectorize(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in feature_names()], dtype=float)


def model_feature_names() -> list[str]:
    """Metric stats + one-hot device type. Device type matters because a
    healthy access-point's baseline latency/jitter looks like a degraded
    core router's -- the model needs type context, not just raw values."""
    return feature_names() + [f"devtype__{t}" for t in DEVICE_TYPES]


def build_model_vector(feats: dict[str, float], device_type: str) -> np.ndarray:
    vec = vectorize(feats)
    onehot = np.array([1.0 if device_type == t else 0.0 for t in DEVICE_TYPES], dtype=float)
    return np.concatenate([vec, onehot])
