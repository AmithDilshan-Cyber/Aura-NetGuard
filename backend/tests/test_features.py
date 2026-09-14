import numpy as np

from backend.app.features import (
    METRICS,
    WINDOW_STEPS,
    build_model_vector,
    extract_features,
    feature_names,
    model_feature_names,
)


def _window(n=WINDOW_STEPS, base=1.0, trend=0.0):
    return [{metric: base + trend * i for metric in METRICS} for i in range(n)]


def test_extract_features_produces_the_declared_schema():
    assert set(extract_features(_window()).keys()) == set(feature_names())


def test_slope_and_delta_detect_an_upward_trend():
    feats = extract_features(_window(trend=2.0))
    for metric in METRICS:
        assert feats[f"{metric}__slope"] > 0
        assert feats[f"{metric}__delta"] > 0


def test_slope_detects_a_downward_trend():
    feats = extract_features(_window(base=50.0, trend=-2.0))
    for metric in METRICS:
        assert feats[f"{metric}__slope"] < 0


def test_flat_series_has_zero_slope_and_std():
    feats = extract_features(_window(trend=0.0))
    for metric in METRICS:
        assert feats[f"{metric}__slope"] == 0
        assert feats[f"{metric}__std"] == 0


def test_build_model_vector_matches_model_feature_names():
    vec = build_model_vector(extract_features(_window()), "core-router")
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == len(model_feature_names())


def test_device_type_onehot_is_exclusive():
    vec = build_model_vector(extract_features(_window()), "firewall")
    onehot = {
        name: value
        for name, value in zip(model_feature_names(), vec)
        if name.startswith("devtype__")
    }
    assert onehot["devtype__firewall"] == 1.0
    assert sum(onehot.values()) == 1.0
