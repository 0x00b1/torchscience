from typing import Tuple

import hypothesis.strategies


@hypothesis.strategies.composite
def complex_avoiding_real_axis(
    draw: hypothesis.strategies.DrawFn,
    real_range: Tuple[float, float] = (-10.0, 10.0),
    min_imag: float = 0.1,
    max_imag: float = 10.0,
) -> complex:
    """Strategy for complex numbers with non-zero imaginary part."""
    real = draw(
        hypothesis.strategies.floats(
            min_value=real_range[0],
            max_value=real_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    imag_magnitude = draw(
        hypothesis.strategies.floats(
            min_value=min_imag,
            max_value=max_imag,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    sign = draw(hypothesis.strategies.sampled_from([-1, 1]))
    return complex(real, sign * imag_magnitude)
