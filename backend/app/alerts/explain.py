"""Turns a raw SHAP-based prediction into a human-centred explanation.

Design goals (this is the research contribution, not just plumbing):
  - Never show a bare probability. Always show *why*.
  - Group the low-level feature attributions (e.g. `cpu_pct__slope`) into an
    operator-relevant root-cause category (resource exhaustion, physical
    link instability, congestion, thermal).
  - Attach an estimated time-to-failure (ETA) computed by extrapolating the
    dominant metric's trend to a critical threshold, so the alert answers
    "how urgent is this" not just "is something wrong".
  - Attach concrete, actionable remediation steps instead of a generic
    "investigate the device" message.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..features import METRIC_META
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

CATEGORY_META = {
    "physical_link_instability": {
        "metrics": {"interface_errors_per_min", "retransmits_per_min"},
        "root_cause": "Likely physical-layer instability (flapping interface, bad cable/transceiver, or duplex mismatch).",
        "actions": [
            "Inspect physical cabling and transceivers on the affected interface.",
            "Check interface CRC/error counters; replace the SFP/cable if errors persist after reseating.",
            "Verify speed/duplex settings match on both ends of the link.",
        ],
    },
    "resource_exhaustion": {
        "metrics": {"cpu_pct", "memory_pct"},
        "root_cause": "Control-plane resource exhaustion (CPU/memory) that can trigger packet drops or process restarts.",
        "actions": [
            "Check for abnormal process/routing-table growth on the device.",
            "Review recent configuration changes or route flaps that could be spiking CPU.",
            "Plan a maintenance-window restart if utilisation does not subside.",
        ],
    },
    "congestion": {
        "metrics": {"latency_ms", "jitter_ms", "bandwidth_util_pct"},
        "root_cause": "Network congestion / queueing delay building up on this path.",
        "actions": [
            "Check for a traffic spike or an unexpected high-bandwidth flow on this link.",
            "Apply QoS/traffic-shaping if utilisation remains elevated.",
            "Confirm there is no routing loop or duplicated traffic inflating load.",
        ],
    },
    "thermal": {
        "metrics": {"temperature_c"},
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
    eta_minutes: float | None


def _dominant_category(contributions: list[FeatureContribution]) -> str:
    # only consider contributions pushing risk *up*
    positive = [c for c in contributions if c.shap_value > 0]
    pool = positive or contributions
    scores: dict[str, float] = {}
    for c in pool:
        metric = _base_metric(c.feature)
        cat = _METRIC_TO_CATEGORY.get(metric)
        if cat is None:
            continue
        scores[cat] = scores.get(cat, 0.0) + abs(c.shap_value)
    if not scores:
        return "congestion"
    return max(scores, key=scores.get)


def _estimate_eta_minutes(window_feats: dict[str, float], category: str) -> float | None:
    candidate_metrics = CATEGORY_META[category]["metrics"]
    best_eta = None
    for metric in candidate_metrics:
        slope = window_feats.get(f"{metric}__slope")
        last = window_feats.get(f"{metric}__last")
        threshold = CRITICAL_THRESHOLDS.get(metric)
        if slope is None or last is None or threshold is None:
            continue
        if slope <= 1e-6 or last >= threshold:
            continue
        steps_needed = (threshold - last) / slope
        eta = steps_needed * STEP_MINUTES
        if eta <= 0:
            continue
        if best_eta is None or eta < best_eta:
            best_eta = eta
    return round(best_eta, 1) if best_eta is not None else None


def explain(prediction: Prediction, window_feats: dict[str, float], device_name: str) -> Explanation:
    category = _dominant_category(prediction.contributions)
    meta = CATEGORY_META[category]

    causes: list[CauseDetail] = []
    for c in prediction.contributions:
        metric = _base_metric(c.feature)
        mm = METRIC_META.get(metric)
        if mm is None or c.shap_value <= 0:
            continue
        if any(existing.metric == metric for existing in causes):
            continue
        causes.append(
            CauseDetail(
                metric=metric,
                label=mm["label"],
                value=round(c.value, 2),
                unit=mm["unit"],
                contribution=round(c.shap_value, 4),
            )
        )
        if len(causes) >= 3:
            break

    eta = _estimate_eta_minutes(window_feats, category)
    eta_phrase = f", est. {eta:.0f} min out" if eta is not None else ""

    if causes:
        lead = causes[0]
        summary = (
            f"{device_name} shows rising {lead.label} ({lead.value}{lead.unit}) consistent with "
            f"{meta['root_cause'].split(' (')[0].lower()}{eta_phrase}."
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
