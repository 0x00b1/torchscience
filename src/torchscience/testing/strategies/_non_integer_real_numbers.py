import math

import hypothesis.strategies


def non_integer_real_numbers(
    min_value: float = -100.0,
    max_value: float = 100.0,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for real numbers that are not integers."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: x != math.floor(x))
