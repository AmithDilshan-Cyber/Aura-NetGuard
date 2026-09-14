from backend.app.alerts.explain import CATEGORY_META, explain
from backend.app.features import METRICS
from backend.app.ml.predictor import FeatureContribution, Prediction


def _prediction(*contributions):
    return Prediction(
        probability=0.8,
        contributions=[FeatureContribution(f, v, s) for f, v, s in contributions],
    )


def _feats(**overrides):
    feats = {}
    for metric in METRICS:
        feats[f"{metric}__last"] = 1.0
        feats[f"{metric}__slope"] = 0.0
    feats.update(overrides)
    return feats


def test_category_is_chosen_from_the_dominant_positive_contributors():
    prediction = _prediction(
        ("cpu_pct__slope", 80.0, 0.30),
        ("memory_pct__last", 85.0, 0.20),
        ("latency_ms__last", 3.0, 0.01),
    )
    assert explain(prediction, _feats(), "core-router-001").category == "resource_exhaustion"


def test_negative_contributions_do_not_become_causes():
    prediction = _prediction(
        ("interface_errors_per_min__last", 4.0, 0.30),
        ("cpu_pct__last", 10.0, -0.50),
    )
    result = explain(prediction, _feats(), "edge-switch-002")
    assert [c.metric for c in result.causes] == ["interface_errors_per_min"]


def test_each_metric_appears_at_most_once_in_causes():
    prediction = _prediction(
        ("cpu_pct__slope", 80.0, 0.30),
        ("cpu_pct__last", 85.0, 0.25),
        ("cpu_pct__max", 88.0, 0.20),
        ("memory_pct__last", 70.0, 0.10),
    )
    metrics = [c.metric for c in explain(prediction, _feats(), "fw-1").causes]
    assert metrics == sorted(set(metrics), key=metrics.index)
    assert metrics.count("cpu_pct") == 1


def test_summary_reads_as_one_clean_sentence():
    """Guards a formatting regression: the category phrase must not carry its
    own full stop into the middle of the sentence."""
    prediction = _prediction(("packet_loss_pct__last", 5.0, 0.4))
    summary = explain(prediction, _feats(**{"packet_loss_pct__slope": 1.0}), "rtr-9").summary
    assert summary == (
        "rtr-9 shows rising packet loss (5.0%) consistent with "
        "congestion and queueing delay on this path, est. 10 min out."
    )


def test_eta_extrapolates_a_rising_metric_to_its_threshold():
    prediction = _prediction(("cpu_pct__slope", 5.0, 0.4))
    # cpu at 80, rising 5 per 30s step, threshold 90 -> 2 steps -> 1 minute
    feats = _feats(**{"cpu_pct__last": 80.0, "cpu_pct__slope": 5.0})
    assert explain(prediction, feats, "rtr-9").eta_minutes == 1.0


def test_implausibly_distant_eta_is_reported_as_unknown():
    """A 5-hour extrapolation from a 5-minute window is not something the
    system should state with a straight face."""
    prediction = _prediction(("cpu_pct__slope", 0.01, 0.4))
    feats = _feats(**{"cpu_pct__last": 10.0, "cpu_pct__slope": 0.01})
    result = explain(prediction, feats, "rtr-9")
    assert result.eta_minutes is None
    assert "est." not in result.summary


def test_eta_is_none_when_nothing_is_trending_up():
    prediction = _prediction(("cpu_pct__last", 80.0, 0.4))
    result = explain(prediction, _feats(**{"cpu_pct__last": 80.0}), "rtr-9")
    assert result.eta_minutes is None
    assert "est." not in result.summary


def test_every_category_carries_actions_and_a_root_cause():
    for category, meta in CATEGORY_META.items():
        assert meta["actions"], f"{category} has no remediation actions"
        assert meta["root_cause"].endswith("."), f"{category} root_cause should be a sentence"
        assert not meta["phrase"].endswith("."), f"{category} phrase must embed mid-sentence"
