"""Vector reflection."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def reflect(direction: Tensor, normal: Tensor) -> Tensor:
    r"""Reflect a direction vector about a surface normal.

    Computes the reflection of a direction vector off a surface defined by its
    normal. This is commonly used in ray tracing, physics simulations, and
    computer graphics for computing reflected rays.

    Mathematical Definition
    -----------------------
    Given direction :math:`D` and normal :math:`N`:

    .. math::
        R = D - 2(D \cdot N)N

    The reflection preserves the component of :math:`D` parallel to the surface
    while reversing the component perpendicular to it.

    Parameters
    ----------
    direction : Tensor, shape (..., 3)
        Direction vectors to reflect. These are typically incident ray directions
        pointing toward the surface (i.e., with a negative component along the normal).

    normal : Tensor, shape (..., 3)
        Surface normal vectors. Should be unit vectors for physically correct
        reflections. If not normalized, the reflection will be scaled.

    Returns
    -------
    Tensor, shape (..., 3)
        Reflected direction vectors. If inputs are unit vectors, the output
        will also be a unit vector.

    Examples
    --------
    Reflect a ray hitting a horizontal surface from above:

    >>> direction = torch.tensor([1.0, -1.0, 0.0])  # coming down at 45 degrees
    >>> normal = torch.tensor([0.0, 1.0, 0.0])       # pointing up
    >>> reflect(direction, normal)
    tensor([1., 1., 0.])

    Reflect a batch of rays:

    >>> directions = torch.randn(100, 3)
    >>> normals = torch.randn(100, 3)
    >>> normals = normals / normals.norm(dim=-1, keepdim=True)  # normalize
    >>> reflected = reflect(directions, normals)

    Notes
    -----
    - The normal should typically be normalized for correct reflection behavior.
    - The direction vector can point either toward or away from the surface;
      the formula works in both cases.
    - This function is differentiable with respect to both inputs.

    See Also
    --------
    refract : Compute refracted rays using Snell's law.

    References
    ----------
    .. [1] Pharr, M., Jakob, W., & Humphreys, G. (2016). Physically Based Rendering:
           From Theory to Implementation (3rd ed.). Morgan Kaufmann.
    """
    if direction.shape[-1] != 3:
        raise ValueError(
            f"reflect: direction must have last dimension 3, got {direction.shape[-1]}"
        )

    if normal.shape[-1] != 3:
        raise ValueError(
            f"reflect: normal must have last dimension 3, got {normal.shape[-1]}"
        )

    return torch.ops.torchscience.reflect(direction, normal)
