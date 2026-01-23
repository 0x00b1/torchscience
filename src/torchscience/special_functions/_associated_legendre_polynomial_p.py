"""Associated Legendre polynomials P_n^m(x)."""

import math

import torch
from torch import Tensor


def associated_legendre_polynomial_p(
    n: int,
    m: int,
    x: Tensor,
    normalized: bool = False,
) -> Tensor:
    r"""Compute associated Legendre polynomial P_n^m(x).

    Mathematical Definition
    -----------------------
    The associated Legendre polynomial is defined as:

    .. math::

        P_n^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_n(x)

    where P_n(x) is the Legendre polynomial of degree n.

    This uses the Condon-Shortley phase convention (the (-1)^m factor).

    For normalized associated Legendre polynomials:

    .. math::

        \tilde{P}_n^m(x) = \sqrt{\frac{(2n+1)(n-m)!}{2(n+m)!}} P_n^m(x)

    which satisfies:

    .. math::

        \int_{-1}^{1} \tilde{P}_n^m(x) \tilde{P}_{n'}^m(x) dx = \delta_{nn'}

    Parameters
    ----------
    n : int
        Degree of the polynomial. Must be non-negative.
    m : int
        Order of the polynomial. Must satisfy |m| <= n.
    x : Tensor
        Input values. Must be in [-1, 1] for real results.
    normalized : bool, optional
        If True, return normalized associated Legendre polynomials.
        Default is False.

    Returns
    -------
    Tensor
        The associated Legendre polynomial P_n^m(x) at each input value.

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> associated_legendre_polynomial_p(2, 0, x)  # P_2^0 = P_2
    tensor([-0.5000, -0.1250,  1.0000])

    >>> associated_legendre_polynomial_p(2, 1, x)
    tensor([-0.0000, -1.2990, -0.0000])

    >>> associated_legendre_polynomial_p(2, 2, x)
    tensor([3.0000, 2.2500, 0.0000])

    Notes
    -----
    Uses three-term recurrence for numerical stability:

    (n-m+1) P_{n+1}^m(x) = (2n+1) x P_n^m(x) - (n+m) P_{n-1}^m(x)

    See Also
    --------
    associated_legendre_polynomial_p_all : Compute all P_n^m for n=0..n_max
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    abs_m = abs(m)
    if abs_m > n:
        raise ValueError(f"|m| must be <= n, got m={m}, n={n}")

    # Handle negative m using symmetry: P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m
    if m < 0:
        result = associated_legendre_polynomial_p(
            n, abs_m, x, normalized=False
        )
        # Apply the symmetry relation
        sign = (-1) ** abs_m
        factor = math.factorial(n - abs_m) / math.factorial(n + abs_m)
        result = sign * factor * result
        if normalized:
            # Apply normalization for negative m
            norm = math.sqrt(
                (2 * n + 1)
                * math.factorial(n - abs_m)
                / (2 * math.factorial(n + abs_m))
            )
            result = norm * result
        return result

    # Now m >= 0
    # Start with P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    # where (2m-1)!! = 1 * 3 * 5 * ... * (2m-1)

    # Compute (1 - x^2)^{1/2}
    sin_theta = torch.sqrt(1 - x * x)

    # P_0^0 = 1
    if m == 0:
        P_mm = torch.ones_like(x)
    else:
        # P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
        double_factorial = 1.0
        for k in range(1, 2 * m, 2):
            double_factorial *= k
        P_mm = ((-1) ** m) * double_factorial * (sin_theta**m)

    if n == m:
        result = P_mm
    else:
        # P_{m+1}^m = x (2m+1) P_m^m
        P_mp1_m = x * (2 * m + 1) * P_mm

        if n == m + 1:
            result = P_mp1_m
        else:
            # Use recurrence: (k-m+1) P_{k+1}^m = (2k+1) x P_k^m - (k+m) P_{k-1}^m
            P_prev = P_mm
            P_curr = P_mp1_m

            for k in range(m + 1, n):
                P_next = ((2 * k + 1) * x * P_curr - (k + m) * P_prev) / (
                    k - m + 1
                )
                P_prev = P_curr
                P_curr = P_next

            result = P_curr

    if normalized:
        # Normalization factor: sqrt((2n+1) * (n-m)! / (2 * (n+m)!))
        norm = math.sqrt(
            (2 * n + 1) * math.factorial(n - m) / (2 * math.factorial(n + m))
        )
        result = norm * result

    return result


def associated_legendre_polynomial_p_all(
    n_max: int,
    x: Tensor,
    normalized: bool = False,
) -> Tensor:
    r"""Compute all associated Legendre polynomials P_n^m(x) for n=0..n_max, m=0..n.

    Parameters
    ----------
    n_max : int
        Maximum degree. Returns polynomials for n=0, 1, ..., n_max.
    x : Tensor
        Input values of shape (...).
    normalized : bool, optional
        If True, return normalized associated Legendre polynomials.
        Default is False.

    Returns
    -------
    Tensor
        Tensor of shape (..., n_max+1, n_max+1) where result[..., n, m] = P_n^m(x).
        For m > n, the values are zero.

    Examples
    --------
    >>> x = torch.tensor([0.5])
    >>> P = associated_legendre_polynomial_p_all(2, x)
    >>> P.shape
    torch.Size([1, 3, 3])
    >>> P[0, 2, 0]  # P_2^0(0.5)
    tensor(-0.1250)
    >>> P[0, 2, 1]  # P_2^1(0.5)
    tensor(-1.2990)
    >>> P[0, 2, 2]  # P_2^2(0.5)
    tensor(2.2500)

    Notes
    -----
    This is more efficient than calling associated_legendre_polynomial_p
    repeatedly, as it uses the recurrence relation to compute all values
    in a single pass.
    """
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative, got {n_max}")

    # Output shape: (..., n_max+1, n_max+1)
    batch_shape = x.shape
    output_shape = batch_shape + (n_max + 1, n_max + 1)
    result = torch.zeros(output_shape, dtype=x.dtype, device=x.device)

    # Compute sin_theta = sqrt(1 - x^2)
    sin_theta = torch.sqrt(1 - x * x)

    # For each m, compute P_m^m, P_{m+1}^m, ..., P_{n_max}^m using recurrence
    for m in range(n_max + 1):
        # Compute P_m^m
        if m == 0:
            P_mm = torch.ones_like(x)
        else:
            # P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
            double_factorial = 1.0
            for k in range(1, 2 * m, 2):
                double_factorial *= k
            P_mm = ((-1) ** m) * double_factorial * (sin_theta**m)

        if normalized:
            norm_mm = math.sqrt(
                (2 * m + 1) * math.factorial(0) / (2 * math.factorial(2 * m))
            )
            result[..., m, m] = norm_mm * P_mm
        else:
            result[..., m, m] = P_mm

        if m < n_max:
            # Compute P_{m+1}^m
            P_mp1_m = x * (2 * m + 1) * P_mm

            if normalized:
                norm_mp1_m = math.sqrt(
                    (2 * (m + 1) + 1)
                    * math.factorial(1)
                    / (2 * math.factorial(2 * m + 1))
                )
                result[..., m + 1, m] = norm_mp1_m * P_mp1_m
            else:
                result[..., m + 1, m] = P_mp1_m

            # Use recurrence for higher n
            P_prev = P_mm
            P_curr = P_mp1_m

            for n in range(m + 2, n_max + 1):
                P_next = ((2 * n - 1) * x * P_curr - (n - 1 + m) * P_prev) / (
                    n - m
                )
                P_prev = P_curr
                P_curr = P_next

                if normalized:
                    norm_nm = math.sqrt(
                        (2 * n + 1)
                        * math.factorial(n - m)
                        / (2 * math.factorial(n + m))
                    )
                    result[..., n, m] = norm_nm * P_curr
                else:
                    result[..., n, m] = P_curr

    return result
