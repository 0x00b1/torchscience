import math

import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DomainError

from ._hermite_polynomial_h import HermitePolynomialH


def hermite_polynomial_h_linspace(
    n: int,
    start: float | None = None,
    end: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate n evenly spaced points for Hermite polynomial evaluation.

    Parameters
    ----------
    n : int
        Number of points.
    start : float
        Start of interval. **Required** for Hermite polynomials (unbounded domain).
    end : float
        End of interval. **Required** for Hermite polynomials (unbounded domain).
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    Tensor
        Evenly spaced points, shape (n,).

    Raises
    ------
    DomainError
        If start or end is not provided (required for unbounded domains).

    Notes
    -----
    Unlike bounded domain polynomials (e.g., Legendre, Chebyshev),
    Hermite polynomials have unbounded domain (-inf, inf).
    Therefore, explicit start and end bounds must be provided.

    For optimal integration/interpolation nodes, use
    hermite_polynomial_h_points instead.

    Examples
    --------
    >>> hermite_polynomial_h_linspace(5, start=-2.0, end=2.0)
    tensor([-2., -1.,  0.,  1.,  2.])

    >>> # This will raise DomainError:
    >>> hermite_polynomial_h_linspace(5)  # Missing start/end
    """
    domain = HermitePolynomialH.DOMAIN

    if math.isinf(domain[0]) or math.isinf(domain[1]):
        if start is None or end is None:
            raise DomainError(
                f"HermitePolynomialH has unbounded domain {domain}. "
                f"Must provide explicit start and end arguments."
            )

    start = start if start is not None else domain[0]
    end = end if end is not None else domain[1]

    return torch.linspace(start, end, n, dtype=dtype, device=device)
