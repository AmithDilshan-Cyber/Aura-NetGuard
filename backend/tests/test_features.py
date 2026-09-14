import numpy as np

from backend.app.features import (
    METRICS,
    WINDOW_STEPS,
    build_model_vector,
    extract_features,
    feature_names,
    model_feature_names,
)


def _fake_window(n=WINDOW_STEPS, base=1.0, trend=0.0):
    return [
        {metric: base + trend * i for metric in METRICS}
        for i in range(n)
    ]


def test_extract_features_shapes():
    window = _fake_window()
    feats = extract_features(window)
    assert set(feats.keys()) == set(feature_names())


def test_slope_detects_upward_trend():
    window = _fake_window(trend=2.0)
    feats = extract_features(window)
    for metric in METRICS:
        assert feats[f"{metric}__slope"] > 0


def test_flat_series_has_zero_slope_and_std():
    window = _fake_window(trend=0.0)
    feats = extract_features(window)
    for metric in METRICS:
        assert feats[f"{metric}__slope"] == 0
        assert feats[f"{metric}__std"] == 0


def test_build_model_vector_length_matches_model_feature_names():
    window = _fake_window()
    feats = extract_features(window)
    vec = build_model_vector(feats, "core-router")
    assert vec.shape[0] == len(model_feature_names())
    assert isinstance(vec, np.ndarray)


def test_device_type_onehot_is_exclusive():
    window = _fake_window()
    feats = extract_features(window)
    vec = build_model_vector(feats, "firewall")
    names = model_feature_names()
    onehot = {n: v for n, v in zip(names, vec) if n.startswith("devtype__")}
    assert onehot["devtype__firewall"] == 1.0
    assert sum(onehot.values()) == 1.0
