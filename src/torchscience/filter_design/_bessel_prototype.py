"""Bessel/Thomson analog lowpass filter prototype."""

import math
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor


def bessel_prototype(
    order: int,
    normalization: Literal["phase", "delay", "magnitude"] = "phase",
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Design an analog Bessel/Thomson lowpass filter prototype.

    Returns the zeros, poles, and gain of a normalized analog Bessel lowpass
    filter. Bessel filters are optimized for maximally flat group delay
    (linear phase in the passband).

    Parameters
    ----------
    order : int
        The order of the filter. Must be positive.
    normalization : {"phase", "delay", "magnitude"}, default "phase"
        Frequency normalization method:
        - "phase": Cutoff is where phase response is -45° × order.
          This is the scipy default.
        - "delay": Cutoff is where group delay drops to 1/sqrt(2) of DC value.
        - "magnitude": Cutoff is where magnitude response is -3 dB.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    zeros : Tensor
        Zeros of the filter (empty for Bessel).
    poles : Tensor
        Poles of the filter, complex tensor of shape (order,).
    gain : Tensor
        System gain, scalar tensor.

    Notes
    -----
    The Bessel filter transfer function is derived from the Bessel polynomial:

    .. math::
        H(s) = \\frac{\\theta_n(0)}{\\theta_n(s)}

    where theta_n(s) is the reverse Bessel polynomial of order n.

    Bessel filters have:
    - Maximally flat group delay in the passband
    - Gradual rolloff (slower than Butterworth, Chebyshev)
    - No passband ripple
    - Excellent step response (minimal overshoot)

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import bessel_prototype
    >>> zeros, poles, gain = bessel_prototype(4)
    >>> poles.shape
    torch.Size([4])
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")
    if normalization not in ("phase", "delay", "magnitude"):
        raise ValueError(
            f"normalization must be 'phase', 'delay', or 'magnitude', got '{normalization}'"
        )

    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    # Determine complex dtype
    if dtype == torch.float32:
        complex_dtype = torch.complex64
    elif dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Get Bessel polynomial coefficients
    # theta_n(s) = sum_{k=0}^{n} a_k * s^k
    # where a_k = (2n-k)! / (2^(n-k) * k! * (n-k)!)
    coeffs = _bessel_polynomial_coefficients(order)

    # Find roots of the Bessel polynomial (these are the poles)
    # Convert to torch tensor for root finding
    coeffs_tensor = torch.tensor(
        coeffs[::-1], dtype=torch.complex128, device=device
    )
    poles = _polynomial_roots(coeffs_tensor)

    # Normalize poles based on normalization type
    if normalization == "phase":
        # Normalize so phase at w=1 is -45° * order
        # This is scipy's default
        norm_factor = _bessel_norm_phase(poles, order)
    elif normalization == "delay":
        # Normalize so group delay at w=0 equals 1
        norm_factor = _bessel_norm_delay(poles)
    else:  # magnitude
        # Normalize so magnitude at w=1 is -3 dB
        norm_factor = _bessel_norm_magnitude(poles)

    poles = poles * norm_factor

    # No zeros for Bessel (all-pole filter)
    zeros = torch.zeros(0, dtype=complex_dtype, device=device)

    # Gain is the product of -poles (to make DC gain = 1)
    gain = torch.prod(-poles).real

    return zeros, poles.to(complex_dtype), gain.to(dtype)


def _bessel_polynomial_coefficients(n: int) -> list:
    """Compute Bessel polynomial coefficients.

    Returns coefficients [a_0, a_1, ..., a_n] for:
    theta_n(s) = sum_{k=0}^{n} a_k * s^k

    Uses the recurrence: a_k = (2n-k)! / (2^(n-k) * k! * (n-k)!)
    """
    coeffs = []
    for k in range(n + 1):
        # a_k = (2n-k)! / (2^(n-k) * k! * (n-k)!)
        numerator = math.factorial(2 * n - k)
        denominator = (
            (2 ** (n - k)) * math.factorial(k) * math.factorial(n - k)
        )
        coeffs.append(numerator / denominator)
    return coeffs


def _polynomial_roots(coeffs: Tensor) -> Tensor:
    """Find roots of polynomial using companion matrix.

    Parameters
    ----------
    coeffs : Tensor
        Polynomial coefficients in descending order [c_n, ..., c_0].

    Returns
    -------
    roots : Tensor
        Roots of the polynomial.
    """
    n = coeffs.numel() - 1
    if n <= 0:
        return torch.zeros(0, dtype=torch.complex128, device=coeffs.device)

    # Normalize to monic polynomial
    coeffs = coeffs / coeffs[0]

    # Build companion matrix
    companion = torch.zeros(
        (n, n), dtype=torch.complex128, device=coeffs.device
    )

    if n > 1:
        companion[1:, :-1] = torch.eye(
            n - 1, dtype=torch.complex128, device=coeffs.device
        )

    companion[:, -1] = -coeffs[1:].flip(0)

    # Eigenvalues are roots
    roots = torch.linalg.eigvals(companion)

    return roots


def _bessel_norm_phase(poles: Tensor, order: int) -> complex:
    """Compute normalization factor for phase normalization.

    Find w such that the phase at w equals -45° * order.
    """
    # For phase normalization, we want phase(H(jw)) = -order * pi/4 at w=1
    # This requires iterative solution, but for standard Bessel polynomials,
    # we can use tabulated values or compute iteratively.

    # Use Newton's method to find the frequency where phase = -order * pi/4
    target_phase = -order * math.pi / 4

    w = 1.0
    for _ in range(50):  # Newton iterations
        # Compute phase and its derivative
        # phase(H(jw)) = -sum(angle(jw - p_k)) for H(s) = k / prod(s - p_k)
        jw = 1j * w
        phase = 0.0
        dphase_dw = 0.0
        for p in poles:
            diff = jw - p
            # Phase contribution
            phase += -torch.angle(diff).item()
            # Derivative: d/dw [-angle(jw - p)] = Re(p) / |jw - p|^2
            p_val = p.item()
            abs_sq = abs(diff.item()) ** 2
            dphase_dw += p_val.real / abs_sq

        error = phase - target_phase
        if abs(error) < 1e-12:
            break

        w = w - error / (dphase_dw + 1e-12)
        w = max(w, 0.01)  # Keep positive

    return 1.0 / w


def _bessel_norm_delay(poles: Tensor) -> complex:
    """Compute normalization factor for delay normalization.

    Normalize so group delay at DC equals 1.
    """
    # Group delay at DC is sum of 1/|p_k| for each pole
    # (since d/dw angle(jw - p) at w=0 is -1/Re(p) for real poles,
    #  or involves more complex formula for complex poles)

    # For Bessel filter with DC gain = 1, the group delay at DC is:
    # tau(0) = sum_k 1/|p_k|^2 * Re(p_k)

    # Actually, for the delay-normalized Bessel filter,
    # the normalization is such that tau(0) = 1
    # This is related to the Bessel polynomial property

    # Use the coefficient-based formula: tau(0) = a_1 / a_0
    # where a_k are Bessel polynomial coefficients
    # For delay normalization, we scale poles by a_0 / a_1

    # Simpler approach: compute group delay at DC and scale
    delay_dc = 0.0
    for p in poles:
        # Group delay contribution from pole p at w=0
        # tau = -d/dw angle(H(jw)) = sum_k d/dw angle(jw - p_k)
        # At w=0: -Im(1/(-p_k)) = -Im(-1/p_k) = Im(1/p_k) = -Im(p_k)/|p_k|^2
        # Wait, let me recalculate...
        # H(s) = prod_k 1/(s-p_k)
        # log H(jw) = -sum_k log(jw - p_k)
        # d/dw log H(jw) = -sum_k 1/(jw - p_k) * j = -j * sum_k 1/(jw - p_k)
        # Group delay = -Im(d/dw log H(jw)) = Re(sum_k 1/(jw - p_k))
        # At w=0: Re(sum_k 1/(-p_k)) = Re(-sum_k 1/p_k)
        # = -sum_k Re(1/p_k) = -sum_k Re(p_k*)/|p_k|^2 = -sum_k Re(p_k)/|p_k|^2
        p_val = p.item()
        delay_dc += -p_val.real / abs(p_val) ** 2

    return complex(delay_dc, 0)


def _bessel_norm_magnitude(poles: Tensor) -> complex:
    """Compute normalization factor for magnitude normalization.

    Normalize so magnitude at w=1 is -3 dB.
    """
    # Find w where |H(jw)|^2 = 0.5
    # Use Newton's method
    # |H(jw)|^2 = k^2 / prod_k |jw - p_k|^2, where k = prod(-poles).real

    # Compute gain k
    k = torch.prod(-poles).real.item()
    k_sq = k**2

    target_mag_sq = 0.5
    w = 1.0

    for _ in range(50):
        jw = 1j * w

        # Compute |H(jw)|^2 = k^2 / prod_k |jw - p_k|^2
        den = 1.0
        dlog_den_dw = 0.0

        for p in poles:
            diff = jw - p
            p_val = p.item()
            abs_sq = abs(diff.item()) ** 2
            den *= abs_sq

            # d/dw |jw - p|^2 = 2 * (w - Im(p))
            # d/dw log(|jw-p|^2) = 2(w - Im(p)) / |jw-p|^2
            dlog_den_dw += 2 * (w - p_val.imag) / abs_sq

        mag_sq = k_sq / den

        # d/dw mag_sq = mag_sq * d/dw log(k^2/den) = -mag_sq * d/dw log(den)
        dmag_sq_dw = -mag_sq * dlog_den_dw

        error = mag_sq - target_mag_sq
        if abs(error) < 1e-12:
            break

        w = w - error / (dmag_sq_dw + 1e-12)
        w = max(w, 0.01)

    return 1.0 / w
