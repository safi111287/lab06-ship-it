"""Tests for validation, probabilities, and voucher decisions."""

import importlib
import json
from pathlib import Path

import joblib
import pytest

import predict

ROOT = Path(__file__).resolve().parents[1]


def test_valid_order_matches_saved_features(valid_order):
    with (ROOT / "models" / "model_info.json").open(encoding="utf-8") as file:
        info = json.load(file)

    expected = info["numeric_features"] + info["categorical_features"]
    validated = predict.validate_order(valid_order)

    assert set(validated) == set(expected)
    assert validated == valid_order


def test_probabilities_are_valid(valid_order):
    result = predict.predict_order(valid_order)

    assert set(result) == {
        "late_probability",
        "on_time_probability",
        "offer_voucher",
        "threshold",
    }
    assert isinstance(result["late_probability"], float)
    assert isinstance(result["on_time_probability"], float)
    assert isinstance(result["offer_voucher"], bool)
    assert isinstance(result["threshold"], float)
    assert 0 <= result["late_probability"] <= 1
    assert 0 <= result["on_time_probability"] <= 1
    assert (
        result["late_probability"] + result["on_time_probability"]
    ) == pytest.approx(1.0)


def test_harder_order_has_higher_late_probability(valid_order):
    easy = {
        **valid_order,
        "distance_km": 1,
        "items": 1,
        "weather": "clear",
        "courier_months": 100,
    }
    hard = {
        **valid_order,
        "distance_km": 25,
        "items": 15,
        "weather": "snow",
        "courier_months": 0,
    }

    easy_result = predict.predict_order(easy)
    hard_result = predict.predict_order(hard)

    assert hard_result["late_probability"] > easy_result["late_probability"]


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.59, 0.60, False),
        (0.60, 0.60, True),
        (0.61, 0.60, True),
        (0.29, 0.30, False),
        (0.30, 0.30, True),
        (0.00, 0.00, True),
        (1.00, 1.00, True),
    ],
)
def test_voucher_threshold_boundary(probability, threshold, expected):
    assert predict.voucher_decision(probability, threshold) is expected


def test_default_threshold_comes_from_saved_metadata(valid_order):
    with (ROOT / "models" / "model_info.json").open(encoding="utf-8") as file:
        saved_threshold = float(json.load(file)["threshold"])

    result = predict.predict_order(valid_order)

    assert result["threshold"] == saved_threshold
    assert predict.voucher_decision(saved_threshold) is True
    assert result["offer_voucher"] is (result["late_probability"] >= saved_threshold)

    if saved_threshold > 0:
        assert predict.voucher_decision(saved_threshold / 2) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_km", -1),
        ("distance_km", 0),
        ("distance_km", 31),
        ("distance_km", None),
        ("distance_km", True),
        ("distance_km", float("nan")),
        ("distance_km", float("inf")),
        ("distance_km", "invalid"),
        ("items", 0),
        ("items", 21),
        ("items", 1.5),
        ("hour", 9),
        ("hour", 23),
        ("hour", 12.5),
        ("courier_months", -1),
        ("courier_months", 121),
    ],
)
def test_invalid_numbers_raise_field_error(valid_order, field, value):
    order = {**valid_order, field: value}

    with pytest.raises(ValueError, match=field):
        predict.predict_order(order)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weather", "storm"),
        ("weather", "Snow"),
        ("weather", ""),
        ("weather", None),
        ("day_type", "holiday"),
        ("vehicle", "truck"),
    ],
)
def test_unknown_categories_raise_field_error(valid_order, field, value):
    order = {**valid_order, field: value}

    with pytest.raises(ValueError, match=field):
        predict.predict_order(order)


@pytest.mark.parametrize(
    "field",
    [
        "distance_km",
        "items",
        "hour",
        "courier_months",
        "weather",
        "day_type",
        "vehicle",
    ],
)
def test_missing_required_fields_raise_error(valid_order, field):
    order = valid_order.copy()
    del order[field]

    with pytest.raises(ValueError, match=field):
        predict.predict_order(order)


@pytest.mark.parametrize("order", [None, [], "invalid", 42])
def test_non_dictionary_orders_raise_error(order):
    with pytest.raises(ValueError, match="order"):
        predict.validate_order(order)


def test_prediction_does_not_modify_input(valid_order):
    original = valid_order.copy()

    predict.predict_order(valid_order)

    assert valid_order == original


def test_model_loads_once_at_import(monkeypatch, valid_order):
    original_load = joblib.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(joblib, "load", tracked_load)
        importlib.reload(predict)

        assert len(calls) == 1

        predict.predict_order(valid_order)
        predict.predict_order(valid_order)

        assert len(calls) == 1
