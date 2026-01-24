"""Conversion from second-order sections to transfer function coefficients."""

from typing import Tuple

import torch
from torch import Tensor

from ._exceptions import SOSNormalizationError


def sos_to_ba(
    sos: Tensor,
    validate: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Convert second-order sections to transfer function coefficients.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    validate : bool, default True
        If True, validate SOS normalization (a0 = 1).

    Returns
    -------
    numerator : Tensor
        Numerator polynomial coefficients in descending order.
    denominator : Tensor
        Denominator polynomial coefficients in descending order.

    Raises
    ------
    SOSNormalizationError
        If validate=True and a0 != 1 for any section.

    Notes
    -----
    Each second-order section represents a biquad filter:

    .. math::
        H_k(z) = \\frac{b_{k0} + b_{k1}z^{-1} + b_{k2}z^{-2}}{a_{k0} + a_{k1}z^{-1} + a_{k2}z^{-2}}

    The overall transfer function is obtained by multiplying all sections:

    .. math::
        H(z) = \\prod_{k=0}^{n-1} H_k(z) = \\frac{B(z)}{A(z)}

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import sos_to_ba
    >>> sos = torch.tensor([[1.0, 2.0, 1.0, 1.0, -0.5, 0.1]])
    >>> b, a = sos_to_ba(sos)
    """
    if sos.numel() == 0:
        return (
            torch.ones(1, dtype=sos.dtype, device=sos.device),
            torch.ones(1, dtype=sos.dtype, device=sos.device),
        )

    n_sections = sos.shape[0]

    # Validate normalization if requested
    if validate:
        a0_vals = sos[:, 3]
        if not torch.allclose(a0_vals, torch.ones_like(a0_vals), atol=1e-10):
            raise SOSNormalizationError(
                f"SOS sections must have a0 = 1, got a0 values: {a0_vals.tolist()}"
            )

    # Start with first section's coefficients
    b = sos[0, :3].clone()
    a = sos[0, 3:].clone()

    # Convolve with each subsequent section
    for i in range(1, n_sections):
        b_section = sos[i, :3]
        a_section = sos[i, 3:]

        b = _convolve_1d(b, b_section)
        a = _convolve_1d(a, a_section)

    return b, a


def _convolve_1d(x: Tensor, y: Tensor) -> Tensor:
    """Convolve two 1D tensors (polynomial multiplication).

    Parameters
    ----------
    x, y : Tensor
        1D tensors to convolve.

    Returns
    -------
    result : Tensor
        Convolution result of length len(x) + len(y) - 1.
    """
    n = x.numel()
    m = y.numel()
    result = torch.zeros(n + m - 1, dtype=x.dtype, device=x.device)

    for i in range(n):
        for j in range(m):
            result[i + j] += x[i] * y[j]

    return result
