"""Loads the trained model and produces explainable predictions.

Every prediction returns not just a probability but a ranked list of which
features pushed the score up (via SHAP), because a bare "87% failure risk"
number is exactly the kind of unexplained alert that causes alert fatigue
and gets ignored by network operators. This module is the bridge between
the black-box classifier and the human-centred alert text generated in
`backend/app/alerts/explain.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap

from ..features import model_feature_names

MODEL_PATH = Path(__file__).parent / "model_store" / "failure_predictor.joblib"


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float  # signed contribution toward the positive (failure) class


@dataclass
class Prediction:
    probability: float
    contributions: list[FeatureContribution]  # sorted by |shap_value| desc


class FailurePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m backend.app.ml.train` first."
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names: list[str] = bundle["feature_names"]
        if self.feature_names != model_feature_names():
            raise ValueError(
                "Feature schema drift: the saved model was trained on a different feature set. "
                "Retrain with `python -m backend.app.ml.train`."
            )
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, vector: np.ndarray, top_k: int = 5) -> Prediction:
        x = vector.reshape(1, -1)
        proba = float(self.model.predict_proba(x)[0, 1])

        values = self.explainer(x).values
        # shap returns (samples, features, classes) for classifiers, or
        # (samples, features) when a single output is produced.
        row = values[0, :, 1] if values.ndim == 3 else values[0]

        order = np.argsort(-np.abs(row))[:top_k]
        contributions = [
            FeatureContribution(
                feature=self.feature_names[i],
                value=float(x[0, i]),
                shap_value=float(row[i]),
            )
            for i in order
        ]
        return Prediction(probability=proba, contributions=contributions)


_predictor: Optional[FailurePredictor] = None


def get_predictor() -> FailurePredictor:
    global _predictor
    if _predictor is None:
        _predictor = FailurePredictor()
    return _predictor
