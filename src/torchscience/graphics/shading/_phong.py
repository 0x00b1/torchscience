"""Phong specular reflectance implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def phong(
    normal: Tensor,
    view: Tensor,
    light: Tensor,
    *,
    shininess: Union[Tensor, float],
) -> Tensor:
    r"""Compute Phong specular reflectance.

    Evaluates the Phong specular reflection model using the reflection
    vector and view direction.

    Mathematical Definition
    -----------------------
    The reflection vector is:

    .. math::
        R = 2(n \cdot l)n - l

    The specular term is:

    .. math::
        S = \max(0, R \cdot v)^{shininess}

    Parameters
    ----------
    normal : Tensor, shape (..., 3)
        Surface normal vectors. Must be normalized.
    view : Tensor, shape (..., 3)
        View direction vectors (toward camera). Must be normalized.
    light : Tensor, shape (..., 3)
        Light direction vectors (toward light). Must be normalized.
    shininess : Tensor or float, shape (...) or scalar
        Specular exponent controlling highlight sharpness.
        Higher values produce smaller, sharper highlights.

    Returns
    -------
    Tensor, shape (...)
        Specular reflectance values in [0, 1].

    Examples
    --------
    >>> normal = torch.tensor([[0.0, 1.0, 0.0]])
    >>> view = torch.tensor([[0.0, 0.707, 0.707]])
    >>> light = torch.tensor([[0.0, 0.707, -0.707]])
    >>> torchscience.graphics.shading.phong(normal, view, light, shininess=32.0)
    tensor([...])

    Notes
    -----
    - All direction vectors must be normalized.
    - Returns 0 when n.l <= 0 (back-facing) or R.v <= 0.

    References
    ----------
    .. [1] B.T. Phong, "Illumination for Computer Generated Pictures",
           Communications of the ACM, 1975.
    """
    if normal.shape[-1] != 3:
        raise ValueError(
            f"normal must have last dimension 3, got {normal.shape[-1]}"
        )
    if view.shape[-1] != 3:
        raise ValueError(
            f"view must have last dimension 3, got {view.shape[-1]}"
        )
    if light.shape[-1] != 3:
        raise ValueError(
            f"light must have last dimension 3, got {light.shape[-1]}"
        )

    if not isinstance(shininess, Tensor):
        shininess = torch.tensor(
            shininess, device=normal.device, dtype=normal.dtype
        )

    return torch.ops.torchscience.phong(normal, view, light, shininess)
