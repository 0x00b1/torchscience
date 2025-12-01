import hypothesis.strategies


def negative_real_numbers(
    min_value: float = -1e10,
    max_value: float = -1e-10,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for negative real numbers."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
