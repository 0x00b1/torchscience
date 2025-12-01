from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation._derivative import derivative


def hessian(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Hessian matrix of a scalar field.

    The Hessian is the matrix of all second partial derivatives:
    H[i,j] = d^2f / (dx_i dx_j).

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute Hessian. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Hessian field with shape (ndim, ndim, *field.shape) if dim is None,
        or (len(dim), len(dim), *field.shape) if dim is specified.

    Examples
    --------
    >>> # Hessian of 2x^2 + 3xy + 4y^2 is [[4, 3], [3, 8]]
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> f = 2*X**2 + 3*X*Y + 4*Y**2
    >>> H = hessian(f, dx=0.05)  # Shape: (2, 2, 21, 21)
    """
    ndim = field.ndim

    # Determine which dimensions to differentiate
    if dim is None:
        spatial_dims = tuple(range(ndim))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Compute all second partial derivatives
    hessian_components = []

    for i, di in enumerate(spatial_dims):
        row = []
        for j, dj in enumerate(spatial_dims):
            if i == j:
                # Diagonal: d^2f/dx_i^2
                h_ij = derivative(
                    field,
                    dim=di,
                    order=2,
                    dx=dx_tuple[i],
                    accuracy=accuracy,
                    kind="central",
                    boundary=boundary,
                )
            elif j > i:
                # Off-diagonal: d^2f/(dx_i dx_j)
                # Compute as d/dx_j (d/dx_i f)
                df_di = derivative(
                    field,
                    dim=di,
                    order=1,
                    dx=dx_tuple[i],
                    accuracy=accuracy,
                    kind="central",
                    boundary=boundary,
                )
                h_ij = derivative(
                    df_di,
                    dim=dj,
                    order=1,
                    dx=dx_tuple[j],
                    accuracy=accuracy,
                    kind="central",
                    boundary=boundary,
                )
            else:
                # Use symmetry: H[i,j] = H[j,i]
                h_ij = hessian_components[j][i]
            row.append(h_ij)
        hessian_components.append(row)

    # Stack into tensor with shape (n_spatial, n_spatial, *field.shape)
    rows = [torch.stack(row, dim=0) for row in hessian_components]
    result = torch.stack(rows, dim=0)

    return result
