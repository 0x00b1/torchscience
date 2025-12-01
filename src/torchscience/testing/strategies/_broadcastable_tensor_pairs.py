from typing import Optional, Tuple

import hypothesis.strategies
import torch

from ._broadcastable_shapes import broadcastable_shapes
from ._tensors import tensors


@hypothesis.strategies.composite
def broadcastable_tensor_pairs(
    draw: hypothesis.strategies.DrawFn,
    dtype: torch.dtype = torch.float64,
    max_dims: int = 4,
    max_side: int = 10,
    elements: Optional[hypothesis.strategies.SearchStrategy[float]] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate pairs of tensors that can be broadcast together."""
    shape1, shape2 = draw(broadcastable_shapes(max_dims, max_side))
    t1 = draw(
        tensors(dtype=dtype, shape=shape1, elements=elements, device=device)
    )
    t2 = draw(
        tensors(dtype=dtype, shape=shape2, elements=elements, device=device)
    )
    return t1, t2
