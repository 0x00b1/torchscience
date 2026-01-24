# src/torchscience/signal_processing/filter/_sosfilt.py
"""Second-order sections filter implementation."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def sosfilt(
    sos: Tensor,
    x: Tensor,
    dim: int = -1,
    zi: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Filter data along one dimension using cascaded second-order sections.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    x : Tensor
        Input signal.
    dim : int
        Dimension along which to filter. Default is -1 (last).
    zi : Tensor, optional
        Initial conditions, shape (n_sections, 2).
        If None, zero initial conditions are used.

    Returns
    -------
    y : Tensor
        Filtered signal, same shape as x.
    zf : Tensor (only if zi is not None)
        Final filter states, shape (n_sections, 2).

    Notes
    -----
    Implements Direct Form II Transposed structure for each section.
    Fully differentiable with respect to both sos and x.
    """
    # Move filter dimension to last
    x = x.movedim(dim, -1)
    original_shape = x.shape
    n_samples = x.shape[-1]

    # Flatten batch dimensions
    x_flat = x.reshape(-1, n_samples)
    batch_size = x_flat.shape[0]

    n_sections = sos.shape[0]

    # Initialize states
    if zi is None:
        states = torch.zeros(
            batch_size, n_sections, 2, dtype=x.dtype, device=x.device
        )
        return_states = False
    else:
        states = zi.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return_states = True

    # Process each section
    y = x_flat
    for section_idx in range(n_sections):
        b0 = sos[section_idx, 0]
        b1 = sos[section_idx, 1]
        b2 = sos[section_idx, 2]
        a0 = sos[section_idx, 3]
        a1 = sos[section_idx, 4]
        a2 = sos[section_idx, 5]

        # Normalize by a0
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        # Get states for this section
        s1 = states[:, section_idx, 0]
        s2 = states[:, section_idx, 1]

        # Output buffer
        y_section = torch.zeros_like(y)

        # Direct Form II Transposed
        for i in range(n_samples):
            x_i = y[:, i]

            # Output
            y_i = b0 * x_i + s1

            # Update states
            s1_new = b1 * x_i - a1 * y_i + s2
            s2_new = b2 * x_i - a2 * y_i

            s1 = s1_new
            s2 = s2_new

            y_section[:, i] = y_i

        # Store final states
        states[:, section_idx, 0] = s1
        states[:, section_idx, 1] = s2

        y = y_section

    # Reshape back
    y = y.reshape(original_shape)
    y = y.movedim(-1, dim)

    if return_states:
        return y, states[0]  # Return states for first batch element
    return y
