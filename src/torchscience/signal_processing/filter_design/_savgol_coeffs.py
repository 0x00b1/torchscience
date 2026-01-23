"""Savitzky-Golay filter coefficient computation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    The Savitzky-Golay filter is a digital filter that can be applied to data
    points for smoothing (or differentiating) without greatly distorting the
    signal. It is based on local polynomial regression.

    Parameters
    ----------
    window_length : int
        The length of the filter window (must be a positive odd integer).
    polyorder : int
        The order of the polynomial used to fit the samples.
        Must be less than window_length.
    deriv : int, optional
        The order of the derivative to compute. Default is 0 (smoothing).
    delta : float, optional
        The spacing of the samples. Default is 1.0.
    pos : int, optional
        Position of the point to evaluate. Default is the middle of the window.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    coeffs : Tensor
        The filter coefficients, shape (window_length,).

    Notes
    -----
    The Savitzky-Golay filter fits a polynomial of degree `polyorder` to the
    data in a sliding window of length `window_length`, then evaluates the
    polynomial (or its derivative) at the center point.

    The coefficients are computed by solving a least-squares problem using
    the Vandermonde matrix approach. For a smoothing filter (deriv=0), the
    coefficients can be interpreted as the weights for a weighted moving
    average.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import savgol_coeffs
    >>> # Smoothing coefficients for window_length=5, polynomial order 2
    >>> coeffs = savgol_coeffs(5, 2)
    >>> coeffs
    tensor([-0.0857,  0.3429,  0.4857,  0.3429, -0.0857], dtype=torch.float64)
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Validate inputs
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError(
            f"window_length must be a positive odd integer, got {window_length}"
        )
    if polyorder < 0:
        raise ValueError(f"polyorder must be non-negative, got {polyorder}")
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder must be less than window_length, got polyorder={polyorder}, "
            f"window_length={window_length}"
        )
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}")
    if deriv > polyorder:
        return torch.zeros(window_length, dtype=dtype, device=device)

    if pos is None:
        pos = window_length // 2

    # Sample positions relative to evaluation point (integer indices)
    # Note: delta is NOT used here, only for final scaling
    x = torch.arange(window_length, dtype=torch.float64, device=device) - pos

    # Build Vandermonde matrix: V[i, j] = x[j]^i for i in [0, polyorder]
    # Each row i corresponds to power i
    powers = torch.arange(polyorder + 1, dtype=torch.float64, device=device)
    V = x.unsqueeze(0) ** powers.unsqueeze(
        1
    )  # Shape: (polyorder+1, window_length)

    # We want to find coefficients c such that V @ c = e_deriv
    # where e_deriv is the standard basis vector with 1 at position deriv
    # This is the least-squares solution to: minimize ||V @ c - e_deriv||
    # The normal equations give: V @ V^T @ beta = V @ e_deriv
    # The pseudo-inverse solution is: c = V^T @ (V @ V^T)^(-1) @ e_deriv

    # Compute V @ V^T (Gram matrix)
    VVT = V @ V.T  # Shape: (polyorder+1, polyorder+1)

    # Solve the system
    VVT_inv = torch.linalg.inv(VVT)
    pseudo_inv = VVT_inv @ V  # Shape: (polyorder+1, window_length)

    # Get coefficients for the derivative
    # The deriv-th row of pseudo_inv gives coefficients for the deriv-th derivative
    coeffs = pseudo_inv[deriv, :]

    # Scale by factorial / delta^deriv
    if deriv > 0:
        factorial = 1.0
        for k in range(1, deriv + 1):
            factorial *= k
        coeffs = coeffs * factorial / (delta**deriv)

    # Reverse the coefficients to match scipy's convention
    # scipy returns coefficients in convolution order (oldest sample first)
    coeffs = coeffs.flip(0)

    return coeffs.to(dtype)
