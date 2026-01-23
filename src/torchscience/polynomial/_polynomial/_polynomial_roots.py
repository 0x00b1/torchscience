import torch
from torch import Tensor

from torchscience.polynomial._degree_error import DegreeError

from ._polynomial import Polynomial

# Threshold for switching from companion matrix to Aberth-Ehrlich
# Companion is O(n^3) for eigenvalue computation, Aberth is O(n^2)
ABERTH_THRESHOLD = 64


def polynomial_roots(p: Polynomial, *, method: str = "auto") -> Tensor:
    """Find polynomial roots via companion matrix eigenvalues or Aberth-Ehrlich.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (..., N).
        Leading coefficient must be non-zero.
    method : str, optional
        Algorithm selection:
        - "auto" (default): Uses companion matrix for degree <= 64,
          Aberth-Ehrlich for higher degrees.
        - "companion": Always use companion matrix eigenvalues (O(n^3)).
        - "aberth": Always use Aberth-Ehrlich iteration (O(n^2)).

    Returns
    -------
    Tensor
        Complex roots, shape (..., N-1). Always complex dtype.

    Raises
    ------
    DegreeError
        If polynomial is constant (degree 0) or zero polynomial.
    ValueError
        If method is not one of "auto", "companion", or "aberth".

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, -3.0, 1.0]))  # (x-1)(x-2)
    >>> polynomial_roots(p)
    tensor([1.+0.j, 2.+0.j])

    >>> # Explicitly use Aberth-Ehrlich
    >>> polynomial_roots(p, method="aberth")
    tensor([2.+0.j, 3.+0.j])

    Notes
    -----
    **Companion matrix method**:
    - Construct companion matrix from normalized coefficients
    - Compute eigenvalues via torch.linalg.eigvals
    - Supports autograd through eigenvalue computation
    - O(n^3) complexity, accurate for low-degree polynomials

    **Aberth-Ehrlich method**:
    - Iterative simultaneous root finding
    - O(n^2) per iteration, typically converges in 10-20 iterations
    - Better scaling for high-degree polynomials (degree > 64)

    For high-degree polynomials (>20), use float64 for accuracy.
    """
    from torchscience.root_finding import aberth_ehrlich

    n = p.shape[-1]  # n = degree + 1
    degree = n - 1

    if n < 2:
        raise DegreeError(
            f"Cannot find roots of constant polynomial (degree 0), got {n} coefficients"
        )

    # Leading coefficient (highest degree)
    leading = p[..., -1]

    # Check for zero leading coefficient
    if torch.any(leading == 0):
        raise DegreeError(
            "Leading coefficient must be non-zero for root finding. "
            "Use polynomial_trim first to remove trailing zeros."
        )

    # Determine method
    if method == "auto":
        method = "companion" if degree <= ABERTH_THRESHOLD else "aberth"

    if method == "aberth":
        # Convert Polynomial to plain Tensor for aberth_ehrlich
        # to avoid Polynomial's __mul__ overriding standard tensor operations
        coeffs = p.as_subclass(Tensor)
        return aberth_ehrlich(coeffs)
    elif method == "companion":
        return _polynomial_roots_companion(p)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'auto', 'companion', or 'aberth'."
        )


def _polynomial_roots_companion(p: Polynomial) -> Tensor:
    """Find polynomial roots using companion matrix eigenvalues.

    This is the internal implementation for the companion matrix method.
    Assumes validation has already been done by the caller.
    """
    n = p.shape[-1]  # n = degree + 1
    leading = p[..., -1]

    # Normalize coefficients by leading coefficient
    # normalized[i] = -coeffs[i] / coeffs[-1]
    normalized = -p[..., :-1] / leading.unsqueeze(-1)

    # Construct companion matrix
    # For polynomial p(x) = a_0 + a_1*x + ... + a_{n-1}*x^{n-1} + x^n (monic)
    # Companion matrix is:
    # [[0, 0, ..., 0, -a_0  ],
    #  [1, 0, ..., 0, -a_1  ],
    #  [0, 1, ..., 0, -a_2  ],
    #  [.                   ],
    #  [0, 0, ..., 1, -a_{n-1}]]
    #
    # The eigenvalues of C are the roots of p(x)

    degree = n - 1  # Degree of polynomial = number of roots

    # Handle batch dimensions
    batch_shape = p.shape[:-1]
    batch_size = batch_shape.numel() if len(batch_shape) > 0 else 1

    # Flatten batch dimensions for construction
    if len(batch_shape) > 0:
        normalized_flat = normalized.reshape(batch_size, degree)
    else:
        normalized_flat = normalized.unsqueeze(0)

    # Build companion matrix
    # Start with zeros
    companion = torch.zeros(
        batch_size, degree, degree, dtype=p.dtype, device=p.device
    )

    # Set subdiagonal to 1
    if degree > 1:
        eye_indices = torch.arange(degree - 1, device=p.device)
        companion[:, eye_indices + 1, eye_indices] = 1.0

    # Set last column to normalized coefficients (negated, already done above)
    companion[:, :, -1] = normalized_flat

    # Compute eigenvalues
    # Convert to complex for eigenvalue computation
    # Preserve precision: float64/complex128 -> complex128, else complex64
    if p.dtype in (torch.float64, torch.complex128):
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64
    companion_complex = companion.to(dtype=complex_dtype)

    roots = torch.linalg.eigvals(companion_complex)

    # Reshape back to batch dimensions
    if len(batch_shape) > 0:
        roots = roots.reshape(*batch_shape, degree)
    else:
        roots = roots.squeeze(0)

    return roots
