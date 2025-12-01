import torch
from torch import Tensor

from ._polynomial import Polynomial, polynomial


def polynomial_from_roots(roots: Tensor) -> Polynomial:
    """Construct monic polynomial from its roots.

    Constructs (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots, shape (..., N). Can be complex.

    Returns
    -------
    Polynomial
        Monic polynomial with given roots, shape (..., N+1).

    Examples
    --------
    >>> roots = torch.tensor([1.0, 2.0])  # (x-1)(x-2) = x^2 - 3x + 2
    >>> p = polynomial_from_roots(roots)
    >>> p.coeffs
    tensor([2., -3., 1.])
    """
    # Build polynomial iteratively: start with (x - r_0), multiply by (x - r_i)

    batch_shape = roots.shape[:-1]
    n_roots = roots.shape[-1]

    if n_roots == 0:
        # Empty roots -> constant polynomial 1
        shape = (*batch_shape, 1) if len(batch_shape) > 0 else (1,)
        return polynomial(
            torch.ones(shape, dtype=roots.dtype, device=roots.device)
        )

    # Start with polynomial (x - r_0) = -r_0 + 1*x
    # coeffs = [-r_0, 1]
    if len(batch_shape) > 0:
        ones = torch.ones(batch_shape, dtype=roots.dtype, device=roots.device)
    else:
        ones = torch.ones((), dtype=roots.dtype, device=roots.device)

    coeffs = torch.stack(
        [
            -roots[..., 0],
            ones,
        ],
        dim=-1,
    )

    # Multiply by (x - r_i) for each remaining root
    for i in range(1, n_roots):
        # Current polynomial has degree i, coeffs has shape (..., i+1)
        # Multiply by (x - r_i) = [-r_i, 1]

        root_i = roots[..., i]

        # (c_0 + c_1*x + ... + c_i*x^i) * (x - r_i)
        # = -r_i*c_0 + (-r_i*c_1 + c_0)*x + (-r_i*c_2 + c_1)*x^2 + ... + c_i*x^{i+1}
        # new_coeffs[0] = -r_i * c_0
        # new_coeffs[j] = -r_i * c_j + c_{j-1} for j = 1..i
        # new_coeffs[i+1] = c_i

        # Shift coefficients (multiply by x)
        shifted = torch.nn.functional.pad(coeffs, (1, 0))  # prepend 0

        # Scale original (multiply by -r_i)
        scaled = torch.nn.functional.pad(coeffs, (0, 1)) * (
            -root_i.unsqueeze(-1)
        )

        coeffs = shifted + scaled

    return Polynomial(coeffs=coeffs)
