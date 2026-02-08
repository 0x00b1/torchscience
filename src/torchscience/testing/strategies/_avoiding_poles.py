import hypothesis.strategies

from ._avoiding_values import avoiding_values


def avoiding_poles(
    max_negative_pole: int = -100,
    min_value: float = -100.0,
    max_value: float = 100.0,
    min_distance: float = 0.01,
) -> hypothesis.strategies.SearchStrategy[float]:
    """Strategy for values avoiding poles at non-positive integers."""
    poles = set(range(max_negative_pole, 1))
    return avoiding_values(poles, min_value, max_value, min_distance)
