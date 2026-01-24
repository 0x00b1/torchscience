"""Double integration using tensor product quadrature."""

from typing import Callable, Tuple, Union

import torch
from torch import Tensor

from torchscience.quadrature._rules import GaussLegendre


def dblquad(
    f: Callable[[Tensor, Tensor], Tensor],
    x_bounds: Tuple[Union[float, Tensor], Union[float, Tensor]],
    y_bounds: Tuple[Union[float, Tensor], Union[float, Tensor]],
    *,
    nx: int = 32,
    ny: int = 32,
) -> Tensor:
    """
    Compute double integral over rectangular domain.

    Computes integral of f(x, y) dx dy using tensor product Gauss-Legendre quadrature.

    Parameters
    ----------
    f : callable
        Integrand f(x, y). Receives meshgrid tensors of shape (*batch, nx, ny).
    x_bounds : tuple
        (x_low, x_high) integration bounds for x.
    y_bounds : tuple
        (y_low, y_high) integration bounds for y.
    nx, ny : int
        Number of quadrature points in each dimension.

    Returns
    -------
    Tensor
        Double integral value.

    Notes
    -----
    Cost is O(nx * ny) function evaluations.
    For high-dimensional integrals, consider Monte Carlo methods.

    Fully differentiable with respect to bounds and closure parameters.

    Uses tensor product of 1D Gauss-Legendre rules, which is exact for
    polynomials of degree <= 2*min(nx, ny) - 1 in each variable.

    Examples
    --------
    >>> # Integrate x*y over unit square
    >>> dblquad(lambda x, y: x * y, (0, 1), (0, 1))  # = 0.25
    """
    x_low, x_high = x_bounds
    y_low, y_high = y_bounds

    # Infer dtype and device
    tensors = [
        t for t in [x_low, x_high, y_low, y_high] if isinstance(t, Tensor)
    ]
    if tensors:
        dtype = tensors[0].dtype
        device = tensors[0].device
    else:
        dtype = torch.float64
        device = torch.device("cpu")

    # Ensure tensors
    if not isinstance(x_low, Tensor):
        x_low = torch.tensor(x_low, dtype=dtype, device=device)
    if not isinstance(x_high, Tensor):
        x_high = torch.tensor(x_high, dtype=dtype, device=device)
    if not isinstance(y_low, Tensor):
        y_low = torch.tensor(y_low, dtype=dtype, device=device)
    if not isinstance(y_high, Tensor):
        y_high = torch.tensor(y_high, dtype=dtype, device=device)

    # Get quadrature rules
    x_rule = GaussLegendre(nx)
    y_rule = GaussLegendre(ny)

    # Get nodes and weights
    x_nodes, x_weights = x_rule.nodes_and_weights(
        x_low, x_high, dtype=dtype, device=device
    )
    y_nodes, y_weights = y_rule.nodes_and_weights(
        y_low, y_high, dtype=dtype, device=device
    )

    # Handle batched vs non-batched
    if x_nodes.dim() == 1 and y_nodes.dim() == 1:
        # Non-batched: create meshgrid
        X, Y = torch.meshgrid(x_nodes, y_nodes, indexing="ij")
        W = torch.outer(x_weights, y_weights)
    else:
        # Batched: need to handle batch dimensions
        # x_nodes: (*batch, nx), y_nodes: (*batch, ny)

        # Determine batch shape from whichever has batch dims
        if x_nodes.dim() > 1:
            batch_shape = x_nodes.shape[:-1]
        else:
            batch_shape = y_nodes.shape[:-1]

        # Expand nodes for meshgrid-like behavior
        if x_nodes.dim() == 1:
            # Expand x_nodes to match batch
            x_nodes = x_nodes.expand(*batch_shape, nx)
            x_weights = x_weights.expand(*batch_shape, nx)
        if y_nodes.dim() == 1:
            # Expand y_nodes to match batch
            y_nodes = y_nodes.expand(*batch_shape, ny)
            y_weights = y_weights.expand(*batch_shape, ny)

        # Create meshgrid in last two dims
        X = x_nodes.unsqueeze(-1).expand(*batch_shape, nx, ny)
        Y = y_nodes.unsqueeze(-2).expand(*batch_shape, nx, ny)

        # Weight tensor product
        W = x_weights.unsqueeze(-1) * y_weights.unsqueeze(
            -2
        )  # (*batch, nx, ny)

    # Evaluate integrand
    values = f(X, Y)  # (*batch, nx, ny)

    # Integrate
    return (values * W).sum(dim=(-2, -1))
