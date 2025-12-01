"""Group delay computation for digital filters."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def group_delay(
    numerator: Tensor,
    denominator: Optional[Tensor] = None,
    n_points: int = 512,
    whole: bool = False,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute the group delay of a digital filter.

    Group delay is the negative derivative of the phase response with respect
    to frequency. For linear phase FIR filters, it equals (N-1)/2 where N is
    the filter length.

    Parameters
    ----------
    numerator : Tensor
        Numerator polynomial coefficients (b) of the transfer function.
    denominator : Tensor, optional
        Denominator polynomial coefficients (a) of the transfer function.
        If not specified, the filter is assumed to be FIR (denominator = [1]).
    n_points : int, optional
        Number of frequency points. Default is 512.
    whole : bool, optional
        If True, compute group delay from 0 to 2*pi (full unit circle).
        If False (default), compute from 0 to pi (Nyquist).
    sampling_frequency : float, optional
        The sampling frequency. If specified, frequencies are returned in Hz.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    frequencies : Tensor
        Frequency points at which group delay is computed. If sampling_frequency
        is specified, these are in Hz; otherwise in radians/sample.
    group_delay : Tensor
        Group delay at each frequency point, in samples.

    Notes
    -----
    The group delay is computed as:

    .. math::
        \\tau_g(\\omega) = -\\frac{d\\phi(\\omega)}{d\\omega}

    where phi(omega) is the phase response.

    For a rational transfer function H(z) = B(z)/A(z), the group delay is:

    .. math::
        \\tau_g = \\text{Re}\\left\\{\\frac{B'(z)}{B(z)} - \\frac{A'(z)}{A(z)}\\right\\}

    where B'(z) and A'(z) are the derivatives with respect to z, evaluated
    on the unit circle.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_analysis import group_delay
    >>> # FIR filter with linear phase has constant group delay
    >>> b = torch.tensor([0.25, 0.5, 0.25])
    >>> w, gd = group_delay(b)
    >>> gd.mean()  # Should be approximately (3-1)/2 = 1
    tensor(1.0000)
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert to float64 for computation
    b = numerator.to(dtype=torch.float64, device=device)

    if denominator is None:
        a = torch.ones(1, dtype=torch.float64, device=device)
    else:
        a = denominator.to(dtype=torch.float64, device=device)

    # Normalize by a[0]
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Frequency points
    if whole:
        max_w = 2 * math.pi
    else:
        max_w = math.pi

    # Use n_points + 1 and exclude endpoint to match scipy convention
    w = torch.linspace(
        0, max_w, n_points + 1, dtype=torch.float64, device=device
    )[:-1]

    # Compute group delay using the derivative method
    # For H(z) = B(z)/A(z), the group delay is:
    # tau = Re{z * B'(z) / B(z)} - Re{z * A'(z) / A(z)}
    # where B'(z) is the derivative with respect to z

    # Create polynomial coefficient indices for derivative
    # d/dz sum(b[k] * z^(-k)) = sum(-k * b[k] * z^(-k-1))
    # Multiply by z: sum(-k * b[k] * z^(-k))
    len_b = len(b)
    len_a = len(a)

    # Coefficient indices (negative powers)
    k_b = torch.arange(len_b, dtype=torch.float64, device=device)
    k_a = torch.arange(len_a, dtype=torch.float64, device=device)

    # z on unit circle: z = exp(j*w)
    z = torch.exp(1j * w)

    # Evaluate polynomials: B(z) = sum(b[k] * z^(-k))
    B = torch.zeros_like(z, dtype=torch.complex128)
    for k in range(len_b):
        B = B + b[k] * z ** (-k)

    # Evaluate A(z)
    A = torch.zeros_like(z, dtype=torch.complex128)
    for k in range(len_a):
        A = A + a[k] * z ** (-k)

    # Evaluate z * B'(z) = sum(-k * b[k] * z^(-k))
    zBprime = torch.zeros_like(z, dtype=torch.complex128)
    for k in range(len_b):
        zBprime = zBprime - k * b[k] * z ** (-k)

    # Evaluate z * A'(z)
    zAprime = torch.zeros_like(z, dtype=torch.complex128)
    for k in range(len_a):
        zAprime = zAprime - k * a[k] * z ** (-k)

    # Group delay
    # tau_g = -d(phase)/d(omega)
    # For H(z) = B(z)/A(z), phase = arg(B) - arg(A)
    # d(arg(B))/d(omega) = Re{z * B'(z) / B(z)}
    # So tau_g = Re{z * A'(z) / A(z)} - Re{z * B'(z) / B(z)}
    # Handle potential division by zero
    eps = 1e-10
    B_safe = torch.where(
        torch.abs(B) < eps,
        torch.complex(torch.tensor(eps), torch.tensor(0.0)),
        B,
    )
    A_safe = torch.where(
        torch.abs(A) < eps,
        torch.complex(torch.tensor(eps), torch.tensor(0.0)),
        A,
    )

    gd = (zAprime / A_safe).real - (zBprime / B_safe).real

    # Convert frequency to Hz if sampling_frequency specified
    if sampling_frequency is not None:
        w = w * sampling_frequency / (2 * math.pi)

    return w.to(dtype), gd.to(dtype)


def group_delay_sos(
    sos: Tensor,
    n_points: int = 512,
    whole: bool = False,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute group delay of a filter in second-order sections format.

    Parameters
    ----------
    sos : Tensor
        Second-order sections representation, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    n_points : int, optional
        Number of frequency points. Default is 512.
    whole : bool, optional
        If True, compute from 0 to 2*pi. If False (default), compute to pi.
    sampling_frequency : float, optional
        The sampling frequency. If specified, frequencies are in Hz.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    frequencies : Tensor
        Frequency points.
    group_delay : Tensor
        Group delay at each frequency point, in samples.

    Notes
    -----
    For cascaded second-order sections, the group delay is the sum of the
    group delays of each section.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_analysis import group_delay_sos
    >>> sos = torch.tensor([[0.25, 0.5, 0.25, 1.0, -0.5, 0.1]])
    >>> w, gd = group_delay_sos(sos)
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    sos = sos.to(dtype=torch.float64, device=device)

    n_sections = sos.shape[0]

    # Initialize total group delay
    total_gd = None
    w = None

    for i in range(n_sections):
        # Extract coefficients
        b = sos[i, :3]
        a = sos[i, 3:]

        # Compute group delay for this section
        w, gd = group_delay(
            b,
            a,
            n_points,
            whole,
            sampling_frequency,
            dtype=dtype,
            device=device,
        )

        if total_gd is None:
            total_gd = gd
        else:
            total_gd = total_gd + gd

    return w, total_gd
