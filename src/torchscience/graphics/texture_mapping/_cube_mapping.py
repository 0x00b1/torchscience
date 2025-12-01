"""Cube mapping texture coordinate implementation."""

from typing import Tuple

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def cube_mapping(
    direction: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Compute cube map face and UV coordinates from a direction vector.

    Maps a 3D direction vector to one of the 6 cube faces and computes
    texture coordinates (u, v) within that face.

    Mathematical Definition
    -----------------------
    Given direction vector (x, y, z), the dominant axis determines the face:
    - Face 0 (+X): x > 0 and |x| >= |y| and |x| >= |z|
    - Face 1 (-X): x < 0 and |x| >= |y| and |x| >= |z|
    - Face 2 (+Y): y > 0 and |y| >= |x| and |y| >= |z|
    - Face 3 (-Y): y < 0 and |y| >= |x| and |y| >= |z|
    - Face 4 (+Z): z > 0 and |z| >= |x| and |z| >= |y|
    - Face 5 (-Z): z < 0 and |z| >= |x| and |z| >= |y|

    UV coordinates are computed by projecting onto the cube face and mapping
    to [0, 1] range.

    Parameters
    ----------
    direction : Tensor
        Direction vectors with shape (..., 3). The last dimension must be 3.
        Does not need to be normalized.

    Returns
    -------
    face : Tensor
        Face indices (0-5) with shape (...). dtype is int64.
    u : Tensor
        U texture coordinates in [0, 1] with shape (...).
    v : Tensor
        V texture coordinates in [0, 1] with shape (...).

    Examples
    --------
    >>> direction = torch.tensor([[1.0, 0.0, 0.0],   # +X
    ...                           [-1.0, 0.0, 0.0],  # -X
    ...                           [0.0, 1.0, 0.0]])  # +Y
    >>> face, u, v = torchscience.graphics.texture_mapping.cube_mapping(direction)
    >>> face
    tensor([0, 1, 2])

    Notes
    -----
    - The direction vector does not need to be normalized. The dominant
      component determines the face.
    - UV coordinates are in [0, 1] with (0.5, 0.5) at the face center.
    - This function is not differentiable due to the discrete face selection.
    """
    return torch.ops.torchscience.cube_mapping(direction)
