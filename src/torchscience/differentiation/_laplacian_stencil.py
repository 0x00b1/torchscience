from __future__ import annotations

from collections import defaultdict

import torch

from torchscience.differentiation._finite_difference_stencil import (
    finite_difference_stencil,
)
from torchscience.differentiation._stencil import FiniteDifferenceStencil


def laplacian_stencil(
    ndim: int,
    accuracy: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> FiniteDifferenceStencil:
    """Generate n-dimensional Laplacian stencil.

    The Laplacian is the sum of second derivatives in each dimension:
    nabla^2 f = d^2f/dx^2 + d^2f/dy^2 + ...

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    accuracy : int
        Accuracy order of the approximation. Default is 2.
    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64.
    device : torch.device, optional
        Output device. Default is CPU.

    Returns
    -------
    FiniteDifferenceStencil
        Laplacian stencil with combined offsets and coefficients.
        For 2D with accuracy=2, this is the 5-point stencil.
        For 3D with accuracy=2, this is the 7-point stencil.

    Examples
    --------
    >>> stencil = laplacian_stencil(ndim=2, accuracy=2)
    >>> stencil.n_points  # 5-point stencil
    5
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Generate 1D second derivative stencil
    stencil_1d = finite_difference_stencil(
        derivative=2,
        accuracy=accuracy,
        kind="central",
        dtype=dtype,
        device=device,
    )

    # Combine 1D stencils embedded in n-D
    # Each dimension contributes offsets along its axis
    offset_to_coeff: defaultdict[tuple, float] = defaultdict(float)

    for dim in range(ndim):
        for i in range(stencil_1d.n_points):
            # Create n-D offset with the 1D offset in position dim
            offset = [0] * ndim
            offset[dim] = stencil_1d.offsets[i, 0].item()
            offset_tuple = tuple(offset)
            offset_to_coeff[offset_tuple] += stencil_1d.coeffs[i].item()

    # Convert to tensors, filtering near-zero coefficients
    offsets_list = []
    coeffs_list = []
    for offset, coeff in offset_to_coeff.items():
        if abs(coeff) > 1e-14:
            offsets_list.append(list(offset))
            coeffs_list.append(coeff)

    offsets_tensor = torch.tensor(
        offsets_list, dtype=torch.int64, device=device
    )
    coeffs_tensor = torch.tensor(coeffs_list, dtype=dtype, device=device)

    # The derivative tuple for Laplacian is (2, 2, ...) representing
    # sum of second derivatives (though this is a sum, not a product)
    derivative_tuple = tuple([2] * ndim)

    return FiniteDifferenceStencil(
        offsets=offsets_tensor,
        coeffs=coeffs_tensor,
        derivative=derivative_tuple,
        accuracy=accuracy,
    )
