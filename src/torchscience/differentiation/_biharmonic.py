from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._laplacian import laplacian


def _biharmonic_impl(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute biharmonic operator of a scalar field.

    The biharmonic is the Laplacian of the Laplacian: nabla^4 f = nabla^2(nabla^2 f).

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute biharmonic. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Biharmonic field with the same shape as the input.

    Examples
    --------
    >>> f = torch.randn(20, 20)
    >>> biharm = biharmonic(f, dx=0.1)  # Shape: (20, 20)
    """
    # Compute Laplacian twice
    lap = laplacian(
        field, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )
    return laplacian(lap, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary)


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def biharmonic(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute biharmonic operator of a scalar field.

    The biharmonic is the Laplacian of the Laplacian: nabla^4 f = nabla^2(nabla^2 f).

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute biharmonic. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Biharmonic field with the same shape as the input.

    Examples
    --------
    >>> f = torch.randn(20, 20)
    >>> biharm = biharmonic(f, dx=0.1)  # Shape: (20, 20)
    """
    return _biharmonic_impl(field, dx, dim, accuracy, boundary)
