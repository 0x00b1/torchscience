from typing import Tuple

import hypothesis.strategies


@hypothesis.strategies.composite
def broadcastable_shapes(
    draw: hypothesis.strategies.DrawFn,
    max_dims: int = 4,
    max_side: int = 10,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Generate pairs of shapes that can be broadcast together."""
    # Generate the result shape first
    result_ndims = draw(
        hypothesis.strategies.integers(min_value=1, max_value=max_dims)
    )
    result_shape = tuple(
        draw(hypothesis.strategies.integers(min_value=1, max_value=max_side))
        for _ in range(result_ndims)
    )

    # Generate first shape by potentially dropping/keeping dimensions
    shape1 = []
    for dim in result_shape:
        choice = draw(
            hypothesis.strategies.sampled_from(["keep", "one", "drop"])
        )
        if choice == "keep":
            shape1.append(dim)
        elif choice == "one":
            shape1.append(1)
        # "drop" means we don't add this dimension

    # Generate second shape similarly
    shape2 = []
    for dim in result_shape:
        choice = draw(
            hypothesis.strategies.sampled_from(["keep", "one", "drop"])
        )
        if choice == "keep":
            shape2.append(dim)
        elif choice == "one":
            shape2.append(1)

    # Ensure at least one element in each shape
    if not shape1:
        shape1 = [1]
    if not shape2:
        shape2 = [1]

    return tuple(shape1), tuple(shape2)
