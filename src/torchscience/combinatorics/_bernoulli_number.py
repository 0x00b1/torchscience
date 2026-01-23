"""Bernoulli numbers B_n."""

from fractions import Fraction
from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor


@lru_cache(maxsize=128)
def _bernoulli_number_exact(n: int) -> Fraction:
    """Compute exact Bernoulli number B_n as a Fraction.

    Uses the Akiyama-Tanigawa algorithm for efficient computation.

    Parameters
    ----------
    n : int
        Index of Bernoulli number. Must be non-negative.

    Returns
    -------
    Fraction
        Exact value of B_n as a fraction.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    # B_0 = 1
    if n == 0:
        return Fraction(1)

    # B_1 = -1/2 (using second Bernoulli number convention)
    if n == 1:
        return Fraction(-1, 2)

    # Odd Bernoulli numbers are 0 for n >= 3
    if n >= 3 and n % 2 == 1:
        return Fraction(0)

    # Akiyama-Tanigawa algorithm
    # Initialize row with A[0,m] = 1/(m+1)
    row = [Fraction(1, m + 1) for m in range(n + 1)]

    # Compute successive rows
    for i in range(1, n + 1):
        for j in range(n - i + 1):
            row[j] = (j + 1) * (row[j] - row[j + 1])

    return row[0]


def bernoulli_number(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Compute Bernoulli number B_n.

    The Bernoulli numbers are defined by the generating function:

    .. math::

        \frac{x}{e^x - 1} = \sum_{n=0}^{\infty} B_n \frac{x^n}{n!}

    This uses the "second Bernoulli number" convention where B_1 = -1/2.

    Mathematical Definition
    -----------------------
    The first several Bernoulli numbers are:

    - B_0 = 1
    - B_1 = -1/2
    - B_2 = 1/6
    - B_3 = 0
    - B_4 = -1/30
    - B_5 = 0
    - B_6 = 1/42

    All odd Bernoulli numbers B_n for n >= 3 are zero.

    Parameters
    ----------
    n : int
        Index of Bernoulli number. Must be non-negative.
    dtype : torch.dtype, optional
        Data type of returned tensor. Default is torch.float64.
    device : torch.device, optional
        Device of returned tensor.

    Returns
    -------
    Tensor
        Scalar tensor containing B_n.

    Examples
    --------
    >>> bernoulli_number(0)
    tensor(1.)

    >>> bernoulli_number(1)
    tensor(-0.5000)

    >>> bernoulli_number(2)
    tensor(0.1667)

    >>> bernoulli_number(4)
    tensor(-0.0333)

    See Also
    --------
    torchscience.polynomial.bernoulli_polynomial_b : Bernoulli polynomials
    """
    if dtype is None:
        dtype = torch.float64

    exact = _bernoulli_number_exact(n)
    value = float(exact)

    return torch.tensor(value, dtype=dtype, device=device)


def bernoulli_number_all(
    n_max: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Compute all Bernoulli numbers B_0, B_1, ..., B_{n_max}.

    Parameters
    ----------
    n_max : int
        Maximum index. Returns n_max + 1 values.
    dtype : torch.dtype, optional
        Data type of returned tensor. Default is torch.float64.
    device : torch.device, optional
        Device of returned tensor.

    Returns
    -------
    Tensor
        1D tensor of shape (n_max + 1,) containing [B_0, B_1, ..., B_{n_max}].

    Examples
    --------
    >>> bernoulli_number_all(6)
    tensor([ 1.0000, -0.5000,  0.1667,  0.0000, -0.0333,  0.0000,  0.0238])
    """
    if dtype is None:
        dtype = torch.float64

    values = [float(_bernoulli_number_exact(i)) for i in range(n_max + 1)]
    return torch.tensor(values, dtype=dtype, device=device)
