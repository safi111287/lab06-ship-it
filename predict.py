"""Load the saved model, validate orders, and make predictions."""

def validate_order(order: dict) -> dict:
    """Return the validated order; raise ValueError for invalid input."""
    raise NotImplementedError


def voucher_decision(
    probability: float,
    threshold: float | None = None,
) -> bool:
    """Return the voucher decision; default to the saved threshold."""
    raise NotImplementedError


def predict_order(order: dict) -> dict:
    """Return these keys:
    late_probability: float
    on_time_probability: float
    offer_voucher: bool
    threshold: float
    """
    raise NotImplementedError