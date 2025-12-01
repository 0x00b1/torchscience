from __future__ import annotations

import torch
from torch import Tensor


def _generate_1d_offsets(
    deriv_order: int,
    accuracy: int,
    kind: str,
    device: torch.device,
) -> Tensor:
    """Generate 1D offset grid for given parameters.

    For central differences, the minimum number of points to achieve
    p-th order accuracy for a d-th derivative is:
        n = d + p - 1 + ((d + p) mod 2)

    This ensures we have an odd number of points (symmetric around center)
    and enough points to cancel the required Taylor series terms.
    """
    if kind == "central":
        # Minimal central stencil: need odd number of points >= deriv_order + 1
        # that achieves the requested accuracy
        n_points = deriv_order + accuracy - 1 + ((deriv_order + accuracy) % 2)
        half = n_points // 2
        offsets = torch.arange(-half, half + 1, device=device)
    elif kind == "forward":
        # Forward differences: need deriv_order + accuracy points starting at 0
        n_points = deriv_order + accuracy
        offsets = torch.arange(0, n_points, device=device)
    else:  # backward
        # Backward differences: need deriv_order + accuracy points ending at 0
        n_points = deriv_order + accuracy
        offsets = torch.arange(-(n_points - 1), 1, device=device)

    return offsets.float()
