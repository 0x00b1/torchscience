import torch
from torch import Tensor

from torchscience.polynomial._degree_error import DegreeError

from ._polynomial import Polynomial


def polynomial_roots(p: Polynomial) -> Tensor:
    """Find polynomial roots via companion matrix eigenvalues.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (..., N).
        Leading coefficient must be non-zero.

    Returns
    -------
    Tensor
        Complex roots, shape (..., N-1). Always complex dtype.

    Raises
    ------
    DegreeError
        If polynomial is constant (degree 0) or zero polynomial.

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, -3.0, 1.0]))  # (x-1)(x-2)
    >>> polynomial_roots(p)
    tensor([1.+0.j, 2.+0.j])

    Notes
    -----
    Uses companion matrix method:
    - Construct companion matrix from normalized coefficients
    - Compute eigenvalues via torch.linalg.eigvals
    - Supports autograd through eigenvalue computation

    For high-degree polynomials (>20), use float64 for accuracy.
    """
    # p IS the coefficient tensor now
    n = p.shape[-1]  # n = degree + 1

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
    companion_complex = companion.to(
        dtype=torch.complex128 if p.dtype == torch.float64 else torch.complex64
    )

    roots = torch.linalg.eigvals(companion_complex)

    # Reshape back to batch dimensions
    if len(batch_shape) > 0:
        roots = roots.reshape(*batch_shape, degree)
    else:
        roots = roots.squeeze(0)

    return roots
