import math

import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DomainError

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_linspace(
    n: int,
    start: float | None = None,
    end: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate n evenly spaced points for Laguerre polynomial evaluation.

    Parameters
    ----------
    n : int
        Number of points.
    start : float
        Start of interval. Required (no default) since domain is unbounded.
    end : float
        End of interval. Required (no default) since domain is unbounded.
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
        If start or end is not provided (required for unbounded domain).

    Notes
    -----
    The natural domain [0, ∞) for Laguerre polynomials is unbounded,
    so explicit start and end values must be provided.

    For non-uniform sampling optimal for integration, use
    laguerre_polynomial_l_points instead.

    Examples
    --------
    >>> laguerre_polynomial_l_linspace(5, start=0.0, end=10.0)
    tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
    """
    domain = LaguerrePolynomialL.DOMAIN

    # For unbounded domain, require explicit bounds
    if math.isinf(domain[1]):
        if start is None or end is None:
            raise DomainError(
                f"LaguerrePolynomialL has unbounded domain [{domain[0]}, {domain[1]}). "
                f"Must provide explicit start and end arguments."
            )

    # Use provided values or domain bounds
    start_val = start if start is not None else domain[0]
    end_val = end if end is not None else domain[1]

    return torch.linspace(start_val, end_val, n, dtype=dtype, device=device)
