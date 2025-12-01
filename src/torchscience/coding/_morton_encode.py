import torch
from torch import Tensor


def morton_encode(coordinates: Tensor) -> Tensor:
    r"""Encode integer coordinates to Morton codes (Z-order curve).

    Morton codes interleave the bits of coordinate values to produce a single
    integer that preserves spatial locality. Points that are close in space
    tend to have similar Morton codes.

    Mathematical Definition
    -----------------------
    For 3D coordinates (x, y, z), the Morton code interleaves bits as:

    .. math::

        \text{morton} = \sum_{i=0}^{20} \left[ x_i \cdot 2^{3i} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i+2} \right]

    where :math:`x_i, y_i, z_i` are the i-th bits of x, y, z respectively.

    For 2D coordinates (x, y), bits are interleaved as:

    .. math::

        \text{morton} = \sum_{i=0}^{31} \left[ x_i \cdot 2^{2i} + y_i \cdot 2^{2i+1} \right]

    Parameters
    ----------
    coordinates : Tensor, shape (..., n_dims)
        Integer coordinates. Supports 2D and 3D:

        - 2D: each coordinate in [0, 2^32)
        - 3D: each coordinate in [0, 2^21)

        The last dimension determines dimensionality (2 or 3).

    Returns
    -------
    Tensor, shape (...), dtype=int64
        Morton codes with interleaved bits.

    Examples
    --------
    3D encoding:

    >>> coords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    >>> morton_encode(coords)
    tensor([0, 1, 2, 3])

    2D encoding:

    >>> coords = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> morton_encode(coords)
    tensor([0, 1, 2, 3])

    Batched encoding:

    >>> coords = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])
    >>> morton_encode(coords)
    tensor([[ 0,  7],
            [56, 63]])

    Notes
    -----
    - Morton codes are also known as Z-order codes
    - The encoding preserves spatial locality: nearby points have similar codes
    - This property makes Morton codes useful for spatial data structures like
      octrees and k-d trees, where cache-efficient traversal is important
    - Negative coordinates are not supported and will produce incorrect results

    See Also
    --------
    morton_decode : Decode Morton codes back to coordinates
    """
    return torch.ops.torchscience.morton_encode(coordinates)
