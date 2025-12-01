from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _gegenbauer_polynomial(n: int, mu: Tensor, x: Tensor) -> Tensor:
    """Evaluate Gegenbauer polynomial C_n^mu(x) using the recurrence relation.

    The Gegenbauer (ultraspherical) polynomials satisfy:
        C_0^mu(x) = 1
        C_1^mu(x) = 2*mu*x
        C_{k+1}^mu(x) = (2*(k+mu)/(k+1)) * x * C_k^mu(x)
                       - ((k+2*mu-1)/(k+1)) * C_{k-1}^mu(x)
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2.0 * mu * x

    c_prev = torch.ones_like(x)  # C_0
    c_curr = 2.0 * mu * x  # C_1

    for k in range(1, n):
        # Compute C_{k+1} from C_k and C_{k-1}
        a_k = 2.0 * (k + mu) / (k + 1.0)
        b_k = (k + 2.0 * mu - 1.0) / (k + 1.0)
        c_next = a_k * x * c_curr - b_k * c_prev
        c_prev = c_curr
        c_curr = c_next

    return c_curr


def ultraspherical_window(
    n: int,
    mu: Union[float, Tensor],
    x_mu: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Ultraspherical (Gegenbauer) window function (symmetric).

    Computes a symmetric ultraspherical window of length n. The ultraspherical
    window is a generalization of the Dolph-Chebyshev window that uses
    Gegenbauer (ultraspherical) polynomials, providing an additional degree
    of freedom to control the window shape.

    Mathematical Definition
    -----------------------
    The ultraspherical window is defined via its frequency response:

        W(theta) = C_{N-1}^{mu}(x_mu * cos(theta)) / C_{N-1}^{mu}(x_mu)

    where C_n^{mu}(x) is the Gegenbauer polynomial of degree n with parameter mu.
    The time-domain window is obtained by inverse FFT of this frequency response.

    Properties
    ----------
    - Generalizes the Dolph-Chebyshev window (mu -> 0 limit)
    - mu controls the trade-off between main lobe width and sidelobe decay
    - x_mu > 1 controls the sidelobe level
    - Larger mu: faster sidelobe decay, wider main lobe
    - Smaller mu: slower sidelobe decay (more uniform), narrower main lobe

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    mu : float or Tensor
        Gegenbauer polynomial parameter (mu > 0).
        Controls the sidelobe roll-off characteristics.
        - mu near 0: approaches Dolph-Chebyshev (equiripple sidelobes)
        - mu = 1: Saramaki window
        - mu = 2: faster sidelobe decay
    x_mu : float or Tensor
        Sidelobe level parameter (x_mu > 1).
        Controls the ratio of main lobe to sidelobe amplitude.
        Larger values give lower sidelobes but wider main lobe.
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
    This function supports autograd - gradients flow through mu and x_mu.

    The window coefficients are computed via:
    1. Evaluate the frequency response at N uniformly spaced points
    2. Apply inverse FFT to obtain time-domain coefficients
    3. Normalize so the center coefficient is 1

    References
    ----------
    .. [1] S. W. A. Bergen and A. Antoniou, "Design of Ultraspherical Window
           Functions with Prescribed Spectral Characteristics," EURASIP Journal
           on Applied Signal Processing, vol. 2004, no. 13, pp. 2053-2065, 2004.

    See Also
    --------
    periodic_ultraspherical_window : Periodic version for spectral analysis.
    dolph_chebyshev_window : Special case when mu -> 0.
    """
    if n < 0:
        raise ValueError(
            f"ultraspherical_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    # Convert parameters to tensors
    target_dtype = dtype or torch.float32

    if not isinstance(mu, Tensor):
        mu = torch.tensor(mu, dtype=target_dtype, device=device)
    if not isinstance(x_mu, Tensor):
        x_mu = torch.tensor(x_mu, dtype=target_dtype, device=device)

    # Ensure consistent dtype
    if mu.dtype != target_dtype:
        mu = mu.to(dtype=target_dtype)
    if x_mu.dtype != target_dtype:
        x_mu = x_mu.to(dtype=target_dtype)

    # Validate parameters
    if mu.item() <= 0:
        raise ValueError(
            f"ultraspherical_window: mu must be positive, got {mu.item()}"
        )
    if x_mu.item() <= 1:
        raise ValueError(
            f"ultraspherical_window: x_mu must be > 1, got {x_mu.item()}"
        )

    return torch.ops.torchscience.ultraspherical_window(
        n, mu, x_mu, dtype, layout, device
    )
