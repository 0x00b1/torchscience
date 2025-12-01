from typing import Optional, Tuple

import hypothesis.extra.numpy
import hypothesis.strategies
import numpy
import torch

from ._shapes import shapes


@hypothesis.strategies.composite
def tensors(
    draw: hypothesis.strategies.DrawFn,
    dtype: torch.dtype = torch.float64,
    shape: Optional[Tuple[int, ...]] = None,
    min_dims: int = 1,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 10,
    elements: Optional[hypothesis.strategies.SearchStrategy[float]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate random tensors with configurable properties."""
    if shape is None:
        shape = draw(shapes(min_dims, max_dims, min_side, max_side))

    if elements is None:
        elements = hypothesis.strategies.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        )

    if dtype in (torch.complex64, torch.complex128):
        # Generate complex tensors
        real_np_dtype = (
            numpy.float32 if dtype == torch.complex64 else numpy.float64
        )
        real_arr = draw(
            hypothesis.extra.numpy.arrays(
                real_np_dtype, shape, elements=elements
            )
        )
        imag_arr = draw(
            hypothesis.extra.numpy.arrays(
                real_np_dtype, shape, elements=elements
            )
        )
        tensor = torch.complex(
            torch.tensor(
                real_arr,
                dtype=torch.float32
                if dtype == torch.complex64
                else torch.float64,
            ),
            torch.tensor(
                imag_arr,
                dtype=torch.float32
                if dtype == torch.complex64
                else torch.float64,
            ),
        )
    else:
        # Generate real tensors
        np_dtype = {
            torch.float16: numpy.float16,
            torch.bfloat16: numpy.float32,  # Use float32 for generation, cast later
            torch.float32: numpy.float32,
            torch.float64: numpy.float64,
        }.get(dtype, numpy.float64)
        arr = draw(
            hypothesis.extra.numpy.arrays(np_dtype, shape, elements=elements)
        )
        tensor = torch.tensor(arr, dtype=dtype)

    return tensor.to(device)
