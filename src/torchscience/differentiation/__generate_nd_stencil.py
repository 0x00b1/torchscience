from __future__ import annotations

from typing import Tuple

import torch

from torchscience.differentiation.__generate_1d_offsets import (
    _generate_1d_offsets,
)
from torchscience.differentiation._fornberg_weights import fornberg_weights
from torchscience.differentiation._stencil import FiniteDifferenceStencil


def _generate_nd_stencil(
    derivative: Tuple[int, ...],
    accuracy: int,
    kind: str,
    dtype: torch.dtype,
    device: torch.device,
) -> FiniteDifferenceStencil:
    """Generate n-dimensional stencil via tensor product."""
    ndim = len(derivative)

    # Generate 1D stencils for each dimension
    stencils_1d = []
    for d in range(ndim):
        deriv_order = derivative[d]
        if deriv_order == 0:
            # Identity in this dimension: single point at offset 0
            stencils_1d.append(
                (
                    torch.tensor([0], device=device),
                    torch.tensor([1.0], device=device),
                )
            )
        else:
            offsets_1d = _generate_1d_offsets(
                deriv_order, accuracy, kind, device
            )
            weights_1d = fornberg_weights(deriv_order, offsets_1d)
            stencils_1d.append((offsets_1d.long(), weights_1d))

    # Compute tensor product
    all_offsets = []
    all_coeffs = []

    # Use recursive Cartesian product
    def cartesian_product(
        dim: int, current_offset: list, current_coeff: float
    ):
        if dim == ndim:
            all_offsets.append(current_offset.copy())
            all_coeffs.append(current_coeff)
            return

        offsets_d, weights_d = stencils_1d[dim]
        for i in range(len(offsets_d)):
            current_offset.append(offsets_d[i].item())
            cartesian_product(
                dim + 1, current_offset, current_coeff * weights_d[i].item()
            )
            current_offset.pop()

    cartesian_product(0, [], 1.0)

    offsets_tensor = torch.tensor(
        all_offsets, dtype=torch.int64, device=device
    )
    coeffs_tensor = torch.tensor(all_coeffs, dtype=dtype, device=device)

    # Filter out near-zero coefficients for sparse representation
    nonzero_mask = coeffs_tensor.abs() > 1e-14
    offsets_tensor = offsets_tensor[nonzero_mask]
    coeffs_tensor = coeffs_tensor[nonzero_mask]

    return FiniteDifferenceStencil(
        offsets=offsets_tensor,
        coeffs=coeffs_tensor,
        derivative=derivative,
        accuracy=accuracy,
    )
