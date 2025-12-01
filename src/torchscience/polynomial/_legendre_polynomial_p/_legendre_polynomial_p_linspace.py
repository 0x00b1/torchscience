import torch
from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP


def legendre_polynomial_p_linspace(
    n: int,
    start: float | None = None,
    end: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate n evenly spaced points in the Legendre domain.

    Parameters
    ----------
    n : int
        Number of points.
    start : float, optional
        Start of interval. Defaults to -1.
    end : float, optional
        End of interval. Defaults to 1.
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    Tensor
        Evenly spaced points, shape (n,).

    Notes
    -----
    The default domain [-1, 1] is the natural domain for Legendre polynomials.
    For non-uniform sampling optimal for integration, use
    legendre_polynomial_p_points instead.

    Examples
    --------
    >>> legendre_polynomial_p_linspace(5)
    tensor([-1.0000, -0.5000,  0.0000,  0.5000,  1.0000])

    >>> legendre_polynomial_p_linspace(3, start=0.0, end=1.0)
    tensor([0.0000, 0.5000, 1.0000])
    """
    domain = LegendrePolynomialP.DOMAIN
    start = start if start is not None else domain[0]
    end = end if end is not None else domain[1]
    return torch.linspace(start, end, n, dtype=dtype, device=device)
