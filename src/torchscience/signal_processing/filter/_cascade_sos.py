"""Utility for cascading multiple SOS filters."""

import torch
from torch import Tensor

from ._exceptions import SOSNormalizationError


def cascade_sos(
    *sos_filters: Tensor,
    validate: bool = True,
) -> Tensor:
    """Cascade multiple SOS filters into a single combined filter.

    Parameters
    ----------
    *sos_filters : Tensor
        Variable number of SOS filter tensors, each with shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    validate : bool, default True
        If True, validate SOS normalization (a0 = 1) for all input filters.

    Returns
    -------
    sos_combined : Tensor
        Combined SOS filter with shape (total_sections, 6), where
        total_sections is the sum of sections from all input filters.

    Raises
    ------
    SOSNormalizationError
        If validate=True and any filter has a0 != 1 for any section.
    ValueError
        If no filters are provided or if filters have incompatible dtypes/devices.

    Notes
    -----
    Cascading SOS filters is equivalent to multiplying their transfer functions:

    .. math::
        H_{combined}(z) = H_1(z) \\cdot H_2(z) \\cdot ... \\cdot H_n(z)

    Since each SOS is already factored into second-order sections, cascading
    simply concatenates the sections. This is more numerically stable than
    converting to BA form and back.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import (
    ...     butterworth_design,
    ...     cascade_sos,
    ... )
    >>> # Create bandpass by cascading lowpass and highpass
    >>> lp = butterworth_design(4, 0.6, filter_type="lowpass")   # (2, 6)
    >>> hp = butterworth_design(4, 0.2, filter_type="highpass")  # (2, 6)
    >>> bp = cascade_sos(lp, hp)  # (4, 6) - equivalent to bandpass
    >>> bp.shape
    torch.Size([4, 6])
    """
    if len(sos_filters) == 0:
        raise ValueError("At least one SOS filter must be provided")

    if len(sos_filters) == 1:
        sos = sos_filters[0]
        if validate and sos.numel() > 0:
            _validate_sos_normalization(sos)
        return sos.clone()

    # Validate all filters and check consistency
    dtype = None
    device = None
    sections_list = []

    for i, sos in enumerate(sos_filters):
        if sos.dim() != 2 or sos.shape[1] != 6:
            raise ValueError(
                f"Filter {i} has invalid shape {sos.shape}. "
                "Expected shape (n_sections, 6)."
            )

        if validate and sos.numel() > 0:
            _validate_sos_normalization(sos, filter_index=i)

        # Track dtype/device from first non-empty filter
        if sos.numel() > 0:
            if dtype is None:
                dtype = sos.dtype
                device = sos.device
            else:
                # Convert to common dtype/device if needed
                if sos.dtype != dtype or sos.device != device:
                    sos = sos.to(dtype=dtype, device=device)

        if sos.shape[0] > 0:
            sections_list.append(sos)

    if len(sections_list) == 0:
        # All filters were empty
        return torch.zeros((0, 6), dtype=dtype or torch.float32, device=device)

    # Concatenate all sections
    return torch.cat(sections_list, dim=0)


def _validate_sos_normalization(sos: Tensor, filter_index: int = 0) -> None:
    """Validate that a0 = 1 for all sections."""
    a0_vals = sos[:, 3]
    if not torch.allclose(a0_vals, torch.ones_like(a0_vals), atol=1e-10):
        raise SOSNormalizationError(
            f"SOS filter {filter_index} has non-normalized sections. "
            f"Expected a0 = 1, got a0 values: {a0_vals.tolist()}. "
            "Use sos_normalize() to normalize before cascading."
        )
