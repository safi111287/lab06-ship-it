"""Shared valid order for prediction tests."""

import pytest


@pytest.fixture
def valid_order():
    """Return a valid order matching the saved model's features."""
    return {
        "distance_km": 5.0,
        "items": 2,
        "hour": 18,
        "courier_months": 12.0,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "bike",
    }
