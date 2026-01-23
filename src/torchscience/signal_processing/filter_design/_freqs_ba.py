"""Analog frequency response from transfer function coefficients."""

from typing import Tuple

import torch
from torch import Tensor


def freqs_ba(
    b: Tensor,
    a: Tensor,
    worN: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute frequency response of an analog filter in BA form.

    Evaluates the transfer function H(s) = B(s) / A(s) at s = j*w for the
    given angular frequencies w, where B and A are polynomials defined by
    their coefficients.

    Parameters
    ----------
    b : Tensor
        Numerator polynomial coefficients (highest power first).
    a : Tensor
        Denominator polynomial coefficients (highest power first).
    worN : Tensor
        Angular frequencies at which to evaluate (rad/s).

    Returns
    -------
    w : Tensor
        The angular frequencies (same as input worN).
    h : Tensor
        Complex frequency response H(jw).

    Notes
    -----
    The transfer function is:

    .. math::
        H(s) = \\frac{b_0 s^n + b_1 s^{n-1} + \\cdots + b_n}
                    {a_0 s^m + a_1 s^{m-1} + \\cdots + a_m}

    where the coefficients are given in descending order of power.

    The evaluation uses Horner's method for numerical stability.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import freqs_ba
    >>> # First-order lowpass: H(s) = 1/(s+1)
    >>> b = torch.tensor([1.0])
    >>> a = torch.tensor([1.0, 1.0])
    >>> w = torch.tensor([0.0, 1.0, 10.0])
    >>> w_out, h = freqs_ba(b, a, w)
    >>> torch.abs(h)
    tensor([1.0000, 0.7071, 0.0995], dtype=torch.float64)

    References
    ----------
    .. [1] scipy.signal.freqs
    """
    # Determine working dtype based on inputs
    if worN.dtype == torch.float32:
        complex_dtype = torch.complex64
        real_dtype = torch.float32
    else:
        complex_dtype = torch.complex128
        real_dtype = torch.float64

    device = worN.device

    # Handle empty frequency array
    if worN.numel() == 0:
        return worN, torch.empty(0, dtype=complex_dtype, device=device)

    # Compute s = j * w
    s = torch.complex(
        torch.zeros_like(worN, dtype=real_dtype),
        worN.to(dtype=real_dtype),
    )  # s = jw

    # Convert coefficients to complex dtype
    b_complex = b.to(dtype=complex_dtype, device=device)
    a_complex = a.to(dtype=complex_dtype, device=device)

    # Evaluate polynomials using Horner's method
    # b[0]*s^n + b[1]*s^(n-1) + ... + b[n]
    # = (...((b[0]*s + b[1])*s + b[2])*s + ...)*s + b[n]
    num = _horner_eval(b_complex, s)
    den = _horner_eval(a_complex, s)

    h = num / den

    return worN, h


def _horner_eval(coeffs: Tensor, x: Tensor) -> Tensor:
    """Evaluate polynomial using Horner's method.

    Parameters
    ----------
    coeffs : Tensor
        Polynomial coefficients in descending order [c_n, c_{n-1}, ..., c_0].
    x : Tensor
        Points at which to evaluate the polynomial.

    Returns
    -------
    Tensor
        Polynomial values at x.
    """
    if coeffs.numel() == 0:
        return torch.ones_like(x)

    result = coeffs[0] * torch.ones_like(x)
    for coef in coeffs[1:]:
        result = result * x + coef

    return result
