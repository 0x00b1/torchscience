from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _chebyshev_polynomial_analytic(order: float, x: Tensor) -> Tensor:
    """Evaluate Chebyshev polynomial T_n(x) using analytic formulas.

    Uses the analytic definitions:
        For |x| <= 1: T_n(x) = cos(n * arccos(x))
        For x > 1:    T_n(x) = cosh(n * arccosh(x))
        For x < -1:   T_n(x) = (-1)^n * cosh(n * arccosh(-x))

    This is more numerically stable than the recurrence relation for large n.
    """
    result = torch.zeros_like(x)

    # |x| <= 1: oscillatory region
    mask_middle = torch.abs(x) <= 1
    if mask_middle.any():
        result[mask_middle] = torch.cos(order * torch.acos(x[mask_middle]))

    # x > 1: exponential growth region
    mask_pos = x > 1
    if mask_pos.any():
        result[mask_pos] = torch.cosh(order * torch.acosh(x[mask_pos]))

    # x < -1: exponential growth with sign
    mask_neg = x < -1
    if mask_neg.any():
        # (-1)^n for continuous n
        sign = torch.cos(
            torch.tensor(torch.pi * order, dtype=x.dtype, device=x.device)
        )
        result[mask_neg] = sign * torch.cosh(order * torch.acosh(-x[mask_neg]))

    return result


def dolph_chebyshev_window(
    n: int,
    attenuation: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Dolph-Chebyshev window function (symmetric).

    Computes a symmetric Dolph-Chebyshev window of length n. The Dolph-Chebyshev
    window (also known as the Chebyshev window) achieves the minimum main lobe
    width for a given sidelobe level, with all sidelobes at equal height
    (equiripple behavior).

    Mathematical Definition
    -----------------------
    The Dolph-Chebyshev window is defined via its frequency response using
    Chebyshev polynomials of the first kind:

        W(theta) = T_{N-1}(x0 * cos(theta / 2)) / T_{N-1}(x0)

    where T_n(x) is the Chebyshev polynomial of degree n, and x0 is derived
    from the desired sidelobe attenuation:

        x0 = cosh(acosh(10^(A/20)) / (N-1))

    The time-domain window is obtained by inverse FFT of this frequency response.

    Properties
    ----------
    - All sidelobes have equal height (equiripple)
    - Optimal: minimizes main lobe width for given sidelobe level
    - Higher attenuation: lower sidelobes but wider main lobe
    - Special case of ultraspherical window with mu -> 0

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    attenuation : float or Tensor
        Desired sidelobe attenuation in decibels (dB). Must be positive.
        Common values:
        - 50 dB: moderate sidelobe suppression
        - 60 dB: good sidelobe suppression
        - 80 dB: excellent sidelobe suppression
        - 100 dB: very high sidelobe suppression
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values.

    Notes
    -----
    This function supports autograd - gradients flow through the attenuation
    parameter.

    The window is computed via:
    1. Compute x0 from the attenuation parameter
    2. Evaluate the frequency response at N uniformly spaced points
    3. Apply inverse FFT to obtain time-domain coefficients
    4. Normalize so the maximum value is 1

    References
    ----------
    .. [1] C. L. Dolph, "A Current Distribution for Broadside Arrays Which
           Optimizes the Relationship between Beam Width and Side-Lobe Level,"
           Proceedings of the IRE, vol. 34, no. 6, pp. 335-348, 1946.
    .. [2] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transform," Proceedings of the IEEE, vol. 66, no. 1,
           pp. 51-83, 1978.

    See Also
    --------
    periodic_dolph_chebyshev_window : Periodic version for spectral analysis.
    ultraspherical_window : Generalization using Gegenbauer polynomials.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.window_function import dolph_chebyshev_window
    >>> window = dolph_chebyshev_window(64, 60.0)
    >>> window.shape
    torch.Size([64])
    """
    if n < 0:
        raise ValueError(
            f"dolph_chebyshev_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    target_dtype = dtype or torch.float32

    if not isinstance(attenuation, Tensor):
        attenuation = torch.tensor(
            attenuation, dtype=target_dtype, device=device
        )

    # Validate attenuation > 0
    if attenuation.item() <= 0:
        raise ValueError(
            f"dolph_chebyshev_window: attenuation must be positive, got {attenuation.item()}"
        )

    return torch.ops.torchscience.dolph_chebyshev_window(
        n, attenuation, dtype, layout, device
    )
