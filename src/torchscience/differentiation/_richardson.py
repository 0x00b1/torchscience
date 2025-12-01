"""Richardson extrapolation for improving finite difference accuracy."""

from __future__ import annotations

from typing import Callable

from torch import Tensor


def richardson_extrapolation(
    f: Callable[[float], Tensor],
    h: float,
    order: int = 2,
    ratio: float = 2.0,
    levels: int = 2,
) -> Tensor:
    """Improve finite difference accuracy using Richardson extrapolation.

    Richardson extrapolation combines estimates at different step sizes
    to cancel leading error terms, achieving higher-order accuracy.

    Uses Neville's algorithm to build the Richardson table:
    1. Compute estimates at step sizes h, h/ratio, h/ratio^2, ...
    2. For each level, combine adjacent entries to cancel error terms
    3. Return the final extrapolated value

    Parameters
    ----------
    f : Callable[[float], Tensor]
        Function f(h) -> Tensor computing the finite difference
        approximation at step size h.
    h : float
        Initial (largest) step size.
    order : int, optional
        Leading error order of the base method. Default is 2 for
        central differences (error is O(h^2)).
    ratio : float, optional
        Step size reduction ratio between levels. Default is 2.0.
    levels : int, optional
        Number of extrapolation levels. Default is 2.

    Returns
    -------
    Tensor
        Extrapolated result with improved accuracy.

    Notes
    -----
    The combination formula at each level is:

        T_{i,j} = (r^p * T_{i+1,j-1} - T_{i,j-1}) / (r^p - 1)

    where:
    - r is the step size ratio
    - p = order + 2*(level-1) is the error order being cancelled
    - T_{i,0} are the initial estimates at step size h/r^i

    For central differences (order=2), the error is O(h^2, h^4, h^6, ...),
    so each level cancels the next even power of h.

    Examples
    --------
    Improve a central difference first derivative approximation:

    >>> import math
    >>> import torch
    >>> x = 1.0
    >>> def central_diff(h):
    ...     return torch.tensor((math.sin(x + h) - math.sin(x - h)) / (2 * h))
    >>> result = richardson_extrapolation(central_diff, h=0.1, order=2, levels=2)
    >>> # result is close to cos(1.0) with much smaller error than central_diff(0.1)

    References
    ----------
    .. [1] Richardson, L. F. (1911). "The Approximate Arithmetical Solution
           by Finite Differences of Physical Problems Involving Differential
           Equations". Philosophical Transactions of the Royal Society A.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if ratio <= 1.0:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    # Compute initial estimates at decreasing step sizes
    # T[i] = f(h / ratio^i) for i = 0, 1, ..., levels-1
    table = []
    current_h = h
    for i in range(levels):
        table.append(f(current_h))
        current_h = current_h / ratio

    # Apply Neville's algorithm (Richardson table)
    # For level j (1 to levels-1), combine adjacent estimates
    # to cancel error term of order p = order + 2*(j-1)
    for j in range(1, levels):
        p = order + 2 * (j - 1)
        r_p = ratio**p

        new_table = []
        for i in range(levels - j):
            # Combine T[i] and T[i+1] to get improved estimate
            # Formula: (r^p * T[i+1] - T[i]) / (r^p - 1)
            improved = (r_p * table[i + 1] - table[i]) / (r_p - 1)
            new_table.append(improved)

        table = new_table

    # Return the final extrapolated value
    return table[0]
