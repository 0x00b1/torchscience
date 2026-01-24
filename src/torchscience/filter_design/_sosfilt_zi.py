"""Compute initial conditions for sosfilt for step response steady-state."""

import torch
from torch import Tensor

from ._lfilter_zi import lfilter_zi


def sosfilt_zi(sos: Tensor) -> Tensor:
    """Compute initial conditions for sosfilt for step response steady-state.

    Computes the initial state for each biquad section such that the output
    of the filter is the steady-state value when the input is a step function
    (all ones).

    This is used to initialize sosfilt to avoid transients when the input
    signal doesn't start at zero.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape ``(n_sections, 6)``.
        Each row is ``[b0, b1, b2, a0, a1, a2]``.

    Returns
    -------
    zi : Tensor
        Initial conditions, shape ``(n_sections, 2)``.

    Raises
    ------
    ValueError
        If sos does not have shape ``(n_sections, 6)``.

    Notes
    -----
    Computes initial conditions for each biquad section by calling
    :func:`lfilter_zi` and cascading through the sections.

    For a filter section with transfer function H(z) = B(z)/A(z), the
    DC gain (gain at zero frequency) is H(1) = sum(b) / sum(a). The
    initial conditions for each section are scaled by the cumulative
    gain of all previous sections to account for the cascaded structure.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import sosfilt_zi
    >>> # Create a simple SOS array (one biquad section)
    >>> sos = torch.tensor([[0.0675, 0.135, 0.0675, 1.0, -0.5, 0.1]],
    ...                    dtype=torch.float64)
    >>> zi = sosfilt_zi(sos)
    """
    # Validate input shape
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("sos must be shape (n_sections, 6)")

    # Convert integer/bool dtypes to float
    if sos.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ):
        sos = sos.to(torch.float64)

    n_sections = sos.shape[0]
    zi = torch.empty((n_sections, 2), dtype=sos.dtype, device=sos.device)

    # Scale factor starts at 1.0
    scale = 1.0

    for section in range(n_sections):
        # Extract b and a coefficients for this section
        b = sos[section, :3]
        a = sos[section, 3:]

        # Compute initial conditions for this section
        # Scale by cumulative gain from previous sections
        zi[section, :] = scale * lfilter_zi(b, a)

        # Update scale with this section's DC gain: H(1) = sum(b) / sum(a)
        scale = scale * torch.sum(b) / torch.sum(a)

    return zi
