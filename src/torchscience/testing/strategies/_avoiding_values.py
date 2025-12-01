from typing import Set

import hypothesis.strategies


def avoiding_values(
    excluded: Set[float],
    min_value: float = -100.0,
    max_value: float = 100.0,
    min_distance: float = 0.01,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values avoiding specified points."""
    return hypothesis.strategies.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: all(abs(x - e) > min_distance for e in excluded))
