"""Turns a raw SHAP-based prediction into a human-centred explanation.

Design goals (this is the research contribution, not just plumbing):
  - Never show a bare probability. Always show *why*.
  - Group the low-level feature attributions (e.g. `cpu_pct__slope`) into an
    operator-relevant root-cause category (resource exhaustion, physical
    link instability, congestion, thermal).
  - Attach an estimated time-to-failure (ETA) computed by extrapolating the
    dominant metric's trend to a critical threshold, so the alert answers
    "how urgent is this" and not just "is something wrong".
  - Attach concrete remediation steps instead of a generic "investigate the
    device" message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..features import METRIC_LABELS
from ..ml.predictor import FeatureContribution, Prediction

CRITICAL_THRESHOLDS = {
    "latency_ms": 50.0,
    "packet_loss_pct": 20.0,
    "jitter_ms": 15.0,
    "bandwidth_util_pct": 95.0,
    "cpu_pct": 90.0,
    "memory_pct": 90.0,
    "interface_errors_per_min": 5.0,
    "temperature_c": 75.0,
    "retransmits_per_min": 5.0,
}

STEP_MINUTES = 0.5  # matches simulator.STEP_SECONDS = 30s

# Beyond this, a linear extrapolation from a 5-minute window is not credible
# and an operator cannot act on it anyway, so we report no ETA instead of a
# confident-looking number we cannot stand behind.
MAX_CREDIBLE_ETA_MINUTES = 60.0

# `phrase` is written to read naturally mid-sentence (no trailing period);
# `root_cause` is the standalone sentence shown as its own field.
CATEGORY_META = {
    "physical_link_instability": {
        "metrics": {"interface_errors_per_min", "retransmits_per_min"},
        "phrase": "physical-layer instability",
        "root_cause": "Likely physical-layer instability (flapping interface, bad cable/transceiver, or duplex mismatch).",
        "actions": [
            "Inspect physical cabling and transceivers on the affected interface.",
            "Check interface CRC/error counters; replace the SFP/cable if errors persist after reseating.",
            "Verify speed/duplex settings match on both ends of the link.",
        ],
    },
    "resource_exhaustion": {
        "metrics": {"cpu_pct", "memory_pct"},
        "phrase": "control-plane resource exhaustion",
        "root_cause": "Control-plane resource exhaustion (CPU/memory) that can trigger packet drops or process restarts.",
        "actions": [
            "Check for abnormal process or routing-table growth on the device.",
            "Review recent configuration changes or route flaps that could be spiking CPU.",
            "Plan a maintenance-window restart if utilisation does not subside.",
        ],
    },
    "congestion": {
        "metrics": {"latency_ms", "jitter_ms", "bandwidth_util_pct", "packet_loss_pct"},
        "phrase": "congestion and queueing delay on this path",
        "root_cause": "Network congestion / queueing delay building up on this path.",
        "actions": [
            "Check for a traffic spike or an unexpected high-bandwidth flow on this link.",
            "Apply QoS/traffic-shaping if utilisation remains elevated.",
            "Confirm there is no routing loop or duplicated traffic inflating load.",
        ],
    },
    "thermal": {
        "metrics": {"temperature_c"},
        "phrase": "thermal stress on the device",
        "root_cause": "Thermal stress on the device -- a cooling or hardware fault risk.",
        "actions": [
            "Check fan status and airflow around the device.",
            "Verify ambient rack/room temperature is within spec.",
            "Schedule a hardware inspection if the condition persists.",
        ],
    },
}

_METRIC_TO_CATEGORY = {m: cat for cat, meta in CATEGORY_META.items() for m in meta["metrics"]}


def _base_metric(feature_name: str) -> str:
    return feature_name.split("__", 1)[0]


@dataclass
class CauseDetail:
    metric: str
    label: str
    value: float
    unit: str
    contribution: float  # positive SHAP contribution toward failure risk


@dataclass
class Explanation:
    category: str
    root_cause: str
    summary: str
    causes: list[CauseDetail]
    recommended_actions: list[str]
    eta_minutes: Optional[float]


def _dominant_category(contributions: list[FeatureContribution]) -> str:
    """Pick the category whose metrics carry the most risk-increasing weight."""
    positive = [c for c in contributions if c.shap_value > 0]
    scores: dict[str, float] = {}
    for c in positive or contributions:
        category = _METRIC_TO_CATEGORY.get(_base_metric(c.feature))
        if category is not None:
            scores[category] = scores.get(category, 0.0) + abs(c.shap_value)
    if not scores:
        return "congestion"
    return max(scores, key=scores.get)


def _estimate_eta_minutes(window_feats: dict[str, float], category: str) -> Optional[float]:
    """Extrapolate the fastest-rising metric in this category to its critical
    threshold. Returns None when nothing in the category is trending upward."""
    best_eta = None
    for metric in CATEGORY_META[category]["metrics"]:
        slope = window_feats.get(f"{metric}__slope")
        last = window_feats.get(f"{metric}__last")
        threshold = CRITICAL_THRESHOLDS.get(metric)
        if slope is None or last is None or threshold is None:
            continue
        if slope <= 1e-6 or last >= threshold:
            continue
        eta = ((threshold - last) / slope) * STEP_MINUTES
        if eta > 0 and (best_eta is None or eta < best_eta):
            best_eta = eta
    if best_eta is None or best_eta > MAX_CREDIBLE_ETA_MINUTES:
        return None
    return round(best_eta, 1)


def _top_causes(contributions: list[FeatureContribution], limit: int = 3) -> list[CauseDetail]:
    causes: list[CauseDetail] = []
    seen: set[str] = set()
    for c in contributions:
        metric = _base_metric(c.feature)
        if c.shap_value <= 0 or metric in seen or metric not in METRIC_LABELS:
            continue
        label, unit = METRIC_LABELS[metric]
        causes.append(
            CauseDetail(
                metric=metric,
                label=label,
                value=round(c.value, 2),
                unit=unit,
                contribution=round(c.shap_value, 4),
            )
        )
        seen.add(metric)
        if len(causes) >= limit:
            break
    return causes


def explain(prediction: Prediction, window_feats: dict[str, float], device_name: str) -> Explanation:
    category = _dominant_category(prediction.contributions)
    meta = CATEGORY_META[category]
    causes = _top_causes(prediction.contributions)
    eta = _estimate_eta_minutes(window_feats, category)

    eta_phrase = f", est. {eta:.0f} min out" if eta is not None else ""
    if causes:
        lead = causes[0]
        summary = (
            f"{device_name} shows rising {lead.label} ({lead.value}{lead.unit}) "
            f"consistent with {meta['phrase']}{eta_phrase}."
        )
    else:
        summary = f"{device_name} is trending toward failure conditions{eta_phrase}."

    return Explanation(
        category=category,
        root_cause=meta["root_cause"],
        summary=summary,
        causes=causes,
        recommended_actions=meta["actions"],
        eta_minutes=eta,
    )
