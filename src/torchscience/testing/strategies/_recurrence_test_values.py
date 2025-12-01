from typing import Tuple

import hypothesis.strategies
import torch


@hypothesis.strategies.composite
def recurrence_test_values(
    draw: hypothesis.strategies.DrawFn,
    base_range: Tuple[float, float] = (-0.99, 0.99),
    max_n: int = 20,
) -> Tuple[torch.Tensor, int]:
    """Generate values for testing recurrence relations."""
    z = draw(
        hypothesis.strategies.floats(
            min_value=base_range[0],
            max_value=base_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    n = draw(hypothesis.strategies.integers(min_value=2, max_value=max_n))
    return torch.tensor([z], dtype=torch.float64), n
