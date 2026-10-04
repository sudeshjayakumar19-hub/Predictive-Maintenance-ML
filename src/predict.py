"""
Prediction and Inference Utilities for Predictive Maintenance Dashboard
Predictive Maintenance ML Capstone Project

This module provides functions to:
- Load the trained champion model bundle (Random Forest).
- Validate and prepare input feature matrices.
- Compute failure-risk probabilities and binary risk predictions.
- Categorize risk levels (LOW, MEDIUM, HIGH) using the calibrated project operating threshold.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Robust relative path handling
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "predictive_maintenance_model.joblib")


def load_model_bundle(model_path: str = _DEFAULT_MODEL_PATH) -> dict:
    """
    Load the persisted model bundle containing:
    - trained model
    - model name
    - feature list
    - evaluation metrics
    - calibrated prediction threshold
    - failure window
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model bundle not found at: {model_path}. "
            "Please ensure 'models/predictive_maintenance_model.joblib' exists or execute train.py first."
        )
    bundle = joblib.load(model_path)
    return bundle


def get_risk_level(prob: float, high_threshold: float = 0.40, med_threshold: float = 0.30) -> str:
    """
    Map failure probability into the 3-tier risk system:
    - HIGH: P >= high_threshold (default 0.40)
    - MEDIUM: med_threshold <= P < high_threshold (0.30 - 0.40)
    - LOW: P < med_threshold (< 0.30)
    """
    if prob >= high_threshold:
        return "HIGH"
    elif prob >= med_threshold:
        return "MEDIUM"
    else:
        return "LOW"


def prepare_input(input_data: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Ensure the input DataFrame contains exactly the required model features
    in the expected column order.
    """
    missing_cols = [c for c in feature_list if c not in input_data.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing {len(missing_cols)} required features: {missing_cols[:5]}...")

    X = input_data[feature_list].copy()
    # Fill any isolated NaNs with 0.0 for safety
    if X.isnull().any().any():
        X = X.fillna(0.0)
    return X


def predict_failure_risk(
    input_data: pd.DataFrame,
    model_bundle: dict = None,
    threshold: float = None
) -> pd.DataFrame:
    """
    Generate failure-risk probabilities and classifications for an input feature matrix.
    
    Returns a DataFrame containing:
    - failure_probability: P(failure within failure_window)
    - predicted_label: 1 if failure_probability >= threshold else 0
    - risk_level: 'LOW', 'MEDIUM', or 'HIGH'
    """
    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle['model']
    feature_list = model_bundle['feature_list']
    operating_threshold = threshold if threshold is not None else model_bundle.get('prediction_threshold', 0.40)

    X = prepare_input(input_data, feature_list)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= operating_threshold).astype(int)
    risk_levels = [get_risk_level(p, high_threshold=operating_threshold) for p in probs]

    results_df = pd.DataFrame({
        'failure_probability': np.round(probs, 4),
        'predicted_label': preds,
        'risk_level': risk_levels
    }, index=input_data.index)

    return results_df
