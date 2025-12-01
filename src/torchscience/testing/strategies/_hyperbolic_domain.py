import hypothesis.strategies


def hyperbolic_domain(
    min_value: float = 1.001,
    max_value: float = 100.0,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values > 1 (hyperbolic continuation domain)."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
