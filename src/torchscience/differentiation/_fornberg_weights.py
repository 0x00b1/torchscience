from __future__ import annotations

import torch
from torch import Tensor

from torchscience.differentiation._exceptions import StencilError


def fornberg_weights(
    derivative_order: int,
    offsets: Tensor,
    x0: float = 0.0,
) -> Tensor:
    """Compute finite difference weights using Fornberg's algorithm.

    This implements the algorithm from:
    Fornberg, B. (1988). "Generation of Finite Difference Formulas on
    Arbitrarily Spaced Grids". Mathematics of Computation, 51(184), 699-706.

    Parameters
    ----------
    derivative_order : int
        Order of derivative (1 for first derivative, 2 for second, etc.).
    offsets : Tensor
        Grid point offsets, shape (n,). Can be non-uniform spacing.
    x0 : float
        Point at which to evaluate the derivative.

    Returns
    -------
    Tensor
        Weights for each grid point, shape (n,).
    """
    n = len(offsets)
    m = derivative_order

    if n <= m:
        raise StencilError(
            f"Need at least {m + 1} points for derivative order {m}, got {n}"
        )

    x = offsets.float() - x0  # Grid points relative to evaluation point
    dtype = offsets.dtype if offsets.is_floating_point() else torch.float64

    # Initialize coefficient table
    # c[j, k] = coefficient for point j when computing k-th derivative
    c = torch.zeros(n, m + 1, dtype=torch.float64, device=offsets.device)

    c[0, 0] = 1.0
    c1 = 1.0

    for j in range(1, n):
        c2 = 1.0
        for k in range(j):
            c3 = x[j] - x[k]
            c2 = c2 * c3

            for s in range(min(j, m), 0, -1):
                c[j, s] = c1 * (s * c[j - 1, s - 1] - x[k] * c[j - 1, s]) / c2

            c[j, 0] = -c1 * x[k] * c[j - 1, 0] / c2

            for s in range(min(j, m), 0, -1):
                c[k, s] = (x[j] * c[k, s] - s * c[k, s - 1]) / c3

            c[k, 0] = x[j] * c[k, 0] / c3

        c1 = c2

    # Extract weights for the requested derivative order
    weights = c[:, m].to(dtype)
    return weights
