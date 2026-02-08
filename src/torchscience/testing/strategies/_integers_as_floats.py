import hypothesis.strategies


def integers_as_floats(
    min_value: int = -100,
    max_value: int = 100,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for integers represented as floats."""
    return hypothesis.strategies.integers(
        min_value=min_value, max_value=max_value
    ).map(float)
