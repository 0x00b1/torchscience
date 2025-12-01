from typing import Tuple

import hypothesis.strategies


@hypothesis.strategies.composite
def shapes(
    draw: hypothesis.strategies.DrawFn,
    min_dims: int = 0,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 10,
) -> Tuple[int, ...]:
    """Strategy for tensor shapes."""
    ndims = draw(
        hypothesis.strategies.integers(min_value=min_dims, max_value=max_dims)
    )
    if ndims == 0:
        return ()
    return tuple(
        draw(
            hypothesis.strategies.integers(
                min_value=min_side, max_value=max_side
            )
        )
        for _ in range(ndims)
    )
