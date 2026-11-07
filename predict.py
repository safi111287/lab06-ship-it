"""Load the saved pipeline, validate orders, and predict late deliveries."""

import json
import math
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "late_orders.joblib"
INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"

with INFO_PATH.open(encoding="utf-8") as file:
    MODEL_INFO = json.load(file)

FEATURE_NAMES = MODEL_INFO["numeric_features"] + MODEL_INFO["categorical_features"]
MODEL = joblib.load(MODEL_PATH)

NUMERIC_RULES = {
    "distance_km": (0.1, 30, False),
    "items": (1, 20, True),
    "hour": (10, 22, True),
    "courier_months": (0, 120, False),
    "prep_minutes": (1, 60, False),
}

CATEGORY_RULES = {
    "weather": ("clear", "rain", "snow"),
    "day_type": ("weekday", "weekend"),
    "vehicle": ("bike", "scooter", "car"),
}


def _unit_interval(value: float, name: str) -> float:
    """Validate a finite number between zero and one."""
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
    """Return a validated order; raise ValueError for invalid input."""
    if not isinstance(order, dict):
        raise ValueError("order must be a dictionary of feature values.")

    missing = [name for name in FEATURE_NAMES if name not in order]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}.")

    extra = [str(name) for name in order if name not in FEATURE_NAMES]
    if extra:
        raise ValueError(f"Unexpected fields: {', '.join(extra)}.")

    validated = {}

    for name in MODEL_INFO["numeric_features"]:
        value = order[name]

        if name == "prep_minutes" and (
            value is None or (isinstance(value, str) and not value.strip())
        ):
            validated[name] = float("nan")
            continue

        minimum, maximum, integer_only = NUMERIC_RULES[name]
        rule = f"{name} must be between {minimum} and {maximum}"
        if integer_only:
            rule += " and must be a whole number"
        rule += "."

        if value is None or isinstance(value, bool):
            raise ValueError(rule)

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(rule) from exc

        if not math.isfinite(number):
            raise ValueError(rule)

        if not minimum <= number <= maximum:
            raise ValueError(rule)

        if integer_only and not number.is_integer():
            raise ValueError(rule)

        validated[name] = int(number) if integer_only else number

    for name in MODEL_INFO["categorical_features"]:
        value = order[name]
        choices = CATEGORY_RULES[name]

        if not isinstance(value, str) or value not in choices:
            raise ValueError(
                f"{name} must be one of: {', '.join(choices)} "
                "(use the exact lowercase spelling)."
            )

        validated[name] = value

    return validated


def voucher_decision(
    probability: float,
    threshold: float | None = None,
) -> bool:
    """Offer a voucher when probability meets the chosen threshold."""
    probability = _unit_interval(probability, "probability")
    if threshold is None:
        threshold = MODEL_INFO["threshold"]

    threshold = _unit_interval(threshold, "threshold")
    return probability >= threshold


def predict_order(order: dict) -> dict:
    """Return probabilities, the voucher decision, and saved threshold."""
    validated = validate_order(order)
    frame = pd.DataFrame([validated], columns=FEATURE_NAMES)

    probabilities = MODEL.predict_proba(frame)[0]
    classes = list(MODEL.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError(
            "The model must use class 0 for on-time orders and class 1 for late orders."
        )

    late_probability = float(probabilities[classes.index(1)])
    on_time_probability = float(probabilities[classes.index(0)])
    threshold = _unit_interval(MODEL_INFO["threshold"], "threshold")

    return {
        "late_probability": late_probability,
        "on_time_probability": on_time_probability,
        "offer_voucher": voucher_decision(late_probability, threshold),
        "threshold": threshold,
    }
