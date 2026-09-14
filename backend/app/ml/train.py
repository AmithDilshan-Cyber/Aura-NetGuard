"""Trains the failure-prediction model and saves it under model_store/.

Usage:
    python -m backend.app.ml.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from .dataset import build_dataset

MODEL_DIR = Path(__file__).parent / "model_store"
MODEL_PATH = MODEL_DIR / "failure_predictor.joblib"
METRICS_PATH = MODEL_DIR / "training_report.json"


def train(
    n_devices: int = 24,
    steps_per_device: int = 4000,
    seed: int = 42,
) -> dict:
    print(f"[train] simulating {n_devices} devices x {steps_per_device} steps ...")
    X, y, groups, feat_names = build_dataset(n_devices=n_devices, steps_per_device=steps_per_device, seed=seed)
    print(f"[train] dataset: {X.shape[0]} rows, {X.shape[1]} features, positive rate={y.mean():.3f}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    report = classification_report(y_test, preds, digits=3, output_dict=True)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print(f"[train] held-out (unseen devices) ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")
    print(classification_report(y_test, preds, digits=3))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feat_names}, MODEL_PATH)

    summary = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "roc_auc_holdout_devices": float(roc_auc),
        "pr_auc_holdout_devices": float(pr_auc),
        "classification_report": report,
        "n_train_devices": int(len(np.unique(groups[train_idx]))),
        "n_test_devices": int(len(np.unique(groups[test_idx]))),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[train] model saved to {MODEL_PATH}")
    print(f"[train] report saved to {METRICS_PATH}")
    return summary


if __name__ == "__main__":
    train()
