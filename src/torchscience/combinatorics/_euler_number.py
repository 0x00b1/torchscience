"""Euler numbers E_n."""

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor


@lru_cache(maxsize=128)
def _euler_number_exact(n: int) -> int:
    """Compute exact Euler number E_n as an integer.

    Uses the recurrence relation for efficient computation.

    Parameters
    ----------
    n : int
        Index of Euler number. Must be non-negative.

    Returns
    -------
    int
        Exact value of E_n.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    # E_0 = 1
    if n == 0:
        return 1

    # Odd Euler numbers are 0
    if n % 2 == 1:
        return 0

    # Use recurrence: E_n = -sum_{k=0}^{n/2-1} C(n, 2k) * E_{2k}
    # for even n >= 2
    from math import comb

    total = 0
    for k in range(n // 2):
        total += comb(n, 2 * k) * _euler_number_exact(2 * k)

    return -total


def euler_number(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Compute Euler number E_n.

    The Euler numbers are defined by the generating function:

    .. math::

        \text{sech}(x) = \frac{2}{e^x + e^{-x}} = \sum_{n=0}^{\infty} E_n \frac{x^n}{n!}

    Mathematical Definition
    -----------------------
    The first several Euler numbers are:

    - E_0 = 1
    - E_1 = 0
    - E_2 = -1
    - E_3 = 0
    - E_4 = 5
    - E_5 = 0
    - E_6 = -61
    - E_8 = 1385

    All odd Euler numbers are zero: E_{2k+1} = 0.

    Parameters
    ----------
    n : int
        Index of Euler number. Must be non-negative.
    dtype : torch.dtype, optional
        Data type of returned tensor. Default is torch.float64.
    device : torch.device, optional
        Device of returned tensor.

    Returns
    -------
    Tensor
        Scalar tensor containing E_n.

    Examples
    --------
    >>> euler_number(0)
    tensor(1.)

    >>> euler_number(2)
    tensor(-1.)

    >>> euler_number(4)
    tensor(5.)

    >>> euler_number(6)
    tensor(-61.)

    See Also
    --------
    torchscience.polynomial.euler_polynomial_e : Euler polynomials
    """
    if dtype is None:
        dtype = torch.float64

    value = _euler_number_exact(n)
    return torch.tensor(float(value), dtype=dtype, device=device)


def euler_number_all(
    n_max: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Compute all Euler numbers E_0, E_1, ..., E_{n_max}.

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
        1D tensor of shape (n_max + 1,) containing [E_0, E_1, ..., E_{n_max}].

    Examples
    --------
    >>> euler_number_all(8)
    tensor([   1.,    0.,   -1.,    0.,    5.,    0.,  -61.,    0., 1385.])
    """
    if dtype is None:
        dtype = torch.float64

    values = [float(_euler_number_exact(i)) for i in range(n_max + 1)]
    return torch.tensor(values, dtype=dtype, device=device)
