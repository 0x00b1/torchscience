import hypothesis.strategies


def chebyshev_domain(
    include_endpoints: bool = False,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values in [-1, 1] (Chebyshev domain)."""
    if include_endpoints:
        return hypothesis.strategies.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    return hypothesis.strategies.floats(
        min_value=-0.999,
        max_value=0.999,
        allow_nan=False,
        allow_infinity=False,
    )
