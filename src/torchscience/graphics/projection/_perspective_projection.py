"""Perspective projection matrix implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def perspective_projection(
    fov: Union[Tensor, float],
    aspect: Union[Tensor, float],
    near: Union[Tensor, float],
    far: Union[Tensor, float],
) -> Tensor:
    r"""Compute a perspective projection matrix.

    Creates a 4x4 perspective projection matrix that transforms points from
    camera/eye space to clip space.

    Mathematical Definition
    -----------------------
    The perspective projection matrix is:

    .. math::
        \begin{bmatrix}
        f/a & 0 & 0 & 0 \\
        0 & f & 0 & 0 \\
        0 & 0 & \frac{f_p + n_p}{n_p - f_p} & \frac{2 f_p n_p}{n_p - f_p} \\
        0 & 0 & -1 & 0
        \end{bmatrix}

    where :math:`f = 1 / \tan(\text{fov}/2)`, :math:`a` is aspect ratio,
    :math:`n_p` is near plane, and :math:`f_p` is far plane.

    Parameters
    ----------
    fov : Tensor or float
        Field of view angle in radians (vertical FOV).
    aspect : Tensor or float
        Aspect ratio (width / height).
    near : Tensor or float
        Distance to near clipping plane (positive).
    far : Tensor or float
        Distance to far clipping plane (positive).

    Returns
    -------
    Tensor
        Perspective projection matrix with shape (..., 4, 4) where ... are
        broadcast batch dimensions from the inputs.

    Examples
    --------
    Create a perspective matrix for 45-degree FOV:

    >>> import math
    >>> M = torchscience.graphics.projection.perspective_projection(
    ...     fov=math.pi / 4,  # 45 degrees
    ...     aspect=16 / 9,
    ...     near=0.1,
    ...     far=100.0,
    ... )
    >>> M.shape
    torch.Size([4, 4])

    Project a point in camera space to clip space:

    >>> point_camera = torch.tensor([1.0, 2.0, -5.0, 1.0])
    >>> point_clip = M @ point_camera
    >>> point_ndc = point_clip[:3] / point_clip[3]  # Perspective divide

    Notes
    -----
    - The camera looks down the negative Z axis (right-handed coordinate system).
    - Near plane maps to Z = -1 in NDC, far plane maps to Z = +1 in NDC.
    - This uses the OpenGL convention for the depth range [-1, 1].
    - All inputs must be positive (fov in (0, pi), aspect > 0, 0 < near < far).
    """
    # Convert scalars to tensors
    if not isinstance(fov, Tensor):
        fov = torch.tensor(fov)
    if not isinstance(aspect, Tensor):
        aspect = torch.tensor(aspect, device=fov.device, dtype=fov.dtype)
    if not isinstance(near, Tensor):
        near = torch.tensor(near, device=fov.device, dtype=fov.dtype)
    if not isinstance(far, Tensor):
        far = torch.tensor(far, device=fov.device, dtype=fov.dtype)

    return torch.ops.torchscience.perspective_projection(
        fov, aspect, near, far
    )
