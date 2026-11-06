"""Load the saved model, validate orders, and make predictions."""

import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "late_orders.joblib"
INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"


@lru_cache(maxsize=1)
def _model_info() -> dict:
    """Read the model's saved configuration."""
    with INFO_PATH.open(encoding="utf-8") as file:
        info = json.load(file)
    if not isinstance(info, dict):
        raise ValueError("model_info.json must contain a JSON object.")
    return info


@lru_cache(maxsize=1)
def _model():
    """Load the trusted, locally saved scikit-learn pipeline once."""
    return joblib.load(MODEL_PATH)


def _threshold() -> float:
    info = _model_info()
    value = info.get("threshold", info.get("best_threshold"))
    if value is None:
        raise ValueError("Saved model_info.json has no threshold.")
    return _unit_interval(value, "Threshold")


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number between 0 and 1.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1.") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1.")
    return number


def validate_order(order: dict) -> dict:
    """Return the validated order; raise ValueError for invalid input."""
    if not isinstance(order, dict) or not order:
        raise ValueError("Order must be a non-empty dictionary.")

    info = _model_info()
    features = info.get("feature_columns", info.get("features"))
    if not isinstance(features, list) or not features or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError("Model metadata must provide a list of features.")

    missing = [feature for feature in features if feature not in order]
    extra = [feature for feature in order if feature not in features]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected fields: {', '.join(extra)}")

    validated = {}
    for feature in features:
        value = order[feature]
        if value is None or isinstance(value, bool):
            raise ValueError(f"Invalid value for {feature}.")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Invalid value for {feature}.")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Invalid value for {feature}.")
        validated[feature] = value
    return validated


def voucher_decision(
    probability: float,
    threshold: float | None = None,
) -> bool:
    """Return the voucher decision; default to the saved threshold."""
    probability = _unit_interval(probability, "Probability")
    cutoff = _threshold() if threshold is None else _unit_interval(
        threshold, "Threshold"
    )
    return probability >= cutoff


def predict_order(order: dict) -> dict:
    """Return these keys:
    late_probability: float
    on_time_probability: float
    offer_voucher: bool
    threshold: float
    """
    validated = validate_order(order)
    model = _model()
    frame = pd.DataFrame([validated])
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model must use class 1 for late deliveries.")

    late_probability = float(probabilities[classes.index(1)])
    on_time_probability = 1.0 - late_probability
    threshold = _threshold()

    return {
        "late_probability": late_probability,
        "on_time_probability": on_time_probability,
        "offer_voucher": voucher_decision(late_probability, threshold),
        "threshold": threshold,
    }