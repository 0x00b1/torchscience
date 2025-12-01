from typing import Tuple

import hypothesis.strategies


@hypothesis.strategies.composite
def complex_numbers(
    draw: hypothesis.strategies.DrawFn,
    real_range: Tuple[float, float] = (-10.0, 10.0),
    imag_range: Tuple[float, float] = (-10.0, 10.0),
) -> complex:
    """Strategy for complex numbers."""
    real = draw(
        hypothesis.strategies.floats(
            min_value=real_range[0],
            max_value=real_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    imag = draw(
        hypothesis.strategies.floats(
            min_value=imag_range[0],
            max_value=imag_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return complex(real, imag)
