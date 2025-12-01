from __future__ import annotations

from collections import defaultdict

import torch

from torchscience.differentiation._laplacian_stencil import laplacian_stencil
from torchscience.differentiation._stencil import FiniteDifferenceStencil


def biharmonic_stencil(
    ndim: int,
    accuracy: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> FiniteDifferenceStencil:
    """Generate biharmonic (nabla^4) stencil.

    The biharmonic operator is the Laplacian applied twice:
    nabla^4 f = nabla^2(nabla^2 f)

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
        Biharmonic stencil computed by self-convolving the Laplacian.
        For 2D with accuracy=2, this is a 13-point stencil.

    Examples
    --------
    >>> stencil = biharmonic_stencil(ndim=2, accuracy=2)
    >>> stencil.n_points  # 13-point stencil
    13
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Get Laplacian stencil
    lap = laplacian_stencil(
        ndim=ndim, accuracy=accuracy, dtype=dtype, device=device
    )

    # Self-convolve: for each pair of offsets, add them and multiply coefficients
    offset_to_coeff: defaultdict[tuple, float] = defaultdict(float)

    for i in range(lap.n_points):
        for j in range(lap.n_points):
            # Sum offsets
            offset_i = lap.offsets[i].tolist()
            offset_j = lap.offsets[j].tolist()
            combined_offset = tuple(a + b for a, b in zip(offset_i, offset_j))

            # Multiply coefficients
            combined_coeff = lap.coeffs[i].item() * lap.coeffs[j].item()
            offset_to_coeff[combined_offset] += combined_coeff

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

    # The derivative tuple for biharmonic is (4, 4, ...) representing
    # the fourth-order operator in each dimension
    derivative_tuple = tuple([4] * ndim)

    return FiniteDifferenceStencil(
        offsets=offsets_tensor,
        coeffs=coeffs_tensor,
        derivative=derivative_tuple,
        accuracy=accuracy,
    )
