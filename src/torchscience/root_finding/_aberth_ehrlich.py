"""Aberth-Ehrlich polynomial root finding algorithm."""

import torch
from torch import Tensor


def _horner_eval(coeffs: Tensor, x: Tensor) -> Tensor:
    """Evaluate polynomial using Horner's method.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients shape (B, N) in ascending order.
    x : Tensor
        Evaluation points shape (B, M).

    Returns
    -------
    Tensor
        Values shape (B, M).
    """
    n = coeffs.shape[-1]
    result = coeffs[..., -1:].expand_as(x)  # Start with leading coefficient

    for i in range(n - 2, -1, -1):
        result = result * x + coeffs[..., i : i + 1]

    return result


def _get_initial_roots(
    coeffs_norm: Tensor, degree: int, device: torch.device, cdtype: torch.dtype
) -> Tensor:
    """Compute initial root guesses using Cauchy bound.

    Uses the Cauchy bound for initial radius and distributes roots
    on a circle with slight offset to avoid symmetry issues.

    Parameters
    ----------
    coeffs_norm : Tensor
        Normalized coefficients (monic polynomial), shape (..., N).
    degree : int
        Polynomial degree.
    device : torch.device
        Device for the output tensor.
    cdtype : torch.dtype
        Complex dtype for the output.

    Returns
    -------
    Tensor
        Initial root guesses, shape (..., degree).
    """
    # Cauchy bound: max(|c_i/c_n|) for i < n, plus 1 for safety
    bound = coeffs_norm[..., :-1].abs().max(dim=-1, keepdim=True)[0] + 1.0

    # Initialize roots on circle with slight perturbation to avoid symmetry
    real_dtype = torch.float64 if cdtype == torch.complex128 else torch.float32
    angles = (
        2
        * torch.pi
        * torch.arange(degree, device=device, dtype=real_dtype)
        / degree
    )
    angles = angles + 0.2  # Slight offset to break symmetry

    # Compute exp(i*angles) = cos(angles) + i*sin(angles)
    z = bound * torch.complex(torch.cos(angles), torch.sin(angles))

    return z


def aberth_ehrlich(
    coeffs: Tensor,
    *,
    maxiter: int = 100,
    tol: float | None = None,
) -> Tensor:
    """Find all roots of a polynomial using Aberth-Ehrlich iteration.

    The Aberth-Ehrlich method is an iterative algorithm that simultaneously
    refines all roots of a polynomial. It is O(n^2) per iteration but typically
    converges in 10-20 iterations, making it O(n^2) total vs O(n^3) for
    companion matrix eigenvalue methods.

    The algorithm combines Newton's method with an Aberth correction term
    that repels roots from each other, preventing multiple roots from
    converging to the same location.

    Parameters
    ----------
    coeffs : Tensor
        Polynomial coefficients in ascending order of powers, shape (..., N).
        Represents c_0 + c_1*x + c_2*x^2 + ... + c_{N-1}*x^{N-1}.
        The polynomial degree is N-1.
    maxiter : int, default=100
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance. Iteration stops when the maximum update
        magnitude is below this threshold. Default is dtype-appropriate:
        1e-12 for float64/complex128, 1e-6 for float32/complex64.

    Returns
    -------
    Tensor
        Complex roots, shape (..., N-1). Always complex even if all roots
        are real. For float64 input, returns complex128; for float32,
        returns complex64. For complex input, preserves dtype.

    Raises
    ------
    ValueError
        If the polynomial has degree < 1 (constant polynomial).

    Examples
    --------
    Find roots of x^2 - 5x + 6 = (x-2)(x-3):

    >>> import torch
    >>> from torchscience.root_finding import aberth_ehrlich
    >>> coeffs = torch.tensor([6.0, -5.0, 1.0], dtype=torch.float64)
    >>> roots = aberth_ehrlich(coeffs)
    >>> sorted(roots.real.tolist())  # doctest: +ELLIPSIS
    [2.0..., 3.0...]

    Find roots of x^2 + 1 = 0 (complex roots +/- i):

    >>> coeffs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    >>> roots = aberth_ehrlich(coeffs)
    >>> sorted(roots.imag.tolist())  # doctest: +ELLIPSIS
    [-1.0..., 1.0...]

    Batched computation:

    >>> coeffs = torch.tensor([
    ...     [2.0, -3.0, 1.0],  # (x-1)(x-2)
    ...     [6.0, -5.0, 1.0],  # (x-2)(x-3)
    ... ], dtype=torch.float64)
    >>> roots = aberth_ehrlich(coeffs)
    >>> roots.shape
    torch.Size([2, 2])

    Notes
    -----
    **Algorithm**: At each iteration, for each root estimate z_k, compute:

    1. Newton step: w_k = p(z_k) / p'(z_k)
    2. Aberth correction: sum_j (1 / (z_k - z_j)) for j != k
    3. Update: z_k <- z_k - w_k / (1 - w_k * correction)

    The Aberth correction prevents roots from colliding by adding
    a repulsive term proportional to the inverse distance to other roots.

    **Convergence**: Typically converges in 10-20 iterations for most
    polynomials. May require more iterations for ill-conditioned
    polynomials with clustered or nearly-repeated roots.

    **Complexity**: O(n^2) per iteration where n is the polynomial degree.
    Total complexity is O(n^2 * iterations), typically O(n^2).

    See Also
    --------
    numpy.polynomial.polynomial.polyroots : NumPy polynomial root finding
    scipy.linalg.eigvals : Companion matrix eigenvalue method

    References
    ----------
    .. [1] O. Aberth, "Iteration methods for finding all zeros of a polynomial
           simultaneously", Mathematics of Computation, 27(122):339-344, 1973.
    .. [2] L.W. Ehrlich, "A modified Newton method for polynomials",
           Communications of the ACM, 10(2):107-108, 1967.
    """
    if tol is None:
        if coeffs.dtype in (torch.float64, torch.complex128):
            tol = 1e-12
        else:
            tol = 1e-6

    # Determine complex dtype for roots
    if coeffs.is_complex():
        cdtype = coeffs.dtype
    else:
        cdtype = (
            torch.complex128
            if coeffs.dtype == torch.float64
            else torch.complex64
        )

    device = coeffs.device
    batch_shape = coeffs.shape[:-1]
    n = coeffs.shape[-1]  # number of coefficients
    degree = n - 1

    if degree < 1:
        raise ValueError("Polynomial must have degree >= 1")

    # Normalize by leading coefficient to make monic
    leading = coeffs[..., -1:].to(cdtype)
    coeffs_norm = coeffs.to(cdtype) / leading

    # Compute derivative coefficients for Newton step
    # d/dx (c_0 + c_1*x + ... + c_n*x^n) = c_1 + 2*c_2*x + ... + n*c_n*x^{n-1}
    # Use real dtype for arange then convert (arange doesn't support complex)
    real_dtype = torch.float64 if cdtype == torch.complex128 else torch.float32
    powers = torch.arange(1, n, device=device, dtype=real_dtype).to(cdtype)
    deriv_coeffs = coeffs_norm[..., 1:] * powers

    # Get initial root guesses
    z = _get_initial_roots(coeffs_norm, degree, device, cdtype)

    # Flatten batch for iteration
    batch_size = batch_shape.numel() if len(batch_shape) > 0 else 1
    z_flat = z.reshape(batch_size, degree)
    coeffs_flat = coeffs_norm.reshape(batch_size, n)
    deriv_flat = deriv_coeffs.reshape(batch_size, degree)

    # Small epsilon for numerical stability
    eps = torch.finfo(
        coeffs.real.dtype if coeffs.is_complex() else coeffs.dtype
    ).eps

    # Aberth-Ehrlich iteration
    for _ in range(maxiter):
        # Evaluate polynomial and derivative at current roots
        p_z = _horner_eval(coeffs_flat, z_flat)  # (B, degree)
        dp_z = _horner_eval(deriv_flat, z_flat)  # (B, degree)

        # Newton correction with stability guard
        w = p_z / (dp_z + eps)

        # Aberth correction: sum of 1/(z_k - z_j) for j != k
        # Use broadcasting: (B, degree, 1) - (B, 1, degree)
        z_diff = z_flat.unsqueeze(-1) - z_flat.unsqueeze(
            -2
        )  # (B, degree, degree)

        # Set diagonal to large value to avoid self-interaction (1/inf = 0)
        mask = torch.eye(degree, device=device, dtype=torch.bool)
        z_diff = z_diff.masked_fill(mask, float("inf"))

        # Sum of 1/(z_k - z_j) for j != k
        correction_sum = (1.0 / z_diff).sum(dim=-1)  # (B, degree)

        # Aberth update: delta = w / (1 - w * correction_sum)
        denominator = 1.0 - w * correction_sum
        # Guard against division by zero
        denominator = torch.where(
            denominator.abs() < eps,
            torch.complex(
                torch.tensor(eps, device=device),
                torch.tensor(0.0, device=device),
            ),
            denominator,
        )
        delta = w / denominator

        z_flat = z_flat - delta

        # Check convergence
        max_delta = delta.abs().max()
        if max_delta < tol:
            break

    # Reshape back to batch shape
    if len(batch_shape) > 0:
        roots = z_flat.reshape(*batch_shape, degree)
    else:
        roots = z_flat.squeeze(0)

    return roots
