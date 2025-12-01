import torch
from torch import Tensor


def morton_decode(codes: Tensor, *, dimensions: int = 3) -> Tensor:
    r"""Decode Morton codes to integer coordinates.

    Extracts the original coordinates from Morton codes by deinterleaving
    the bits. This is the inverse operation of :func:`morton_encode`.

    Mathematical Definition
    -----------------------
    For 3D Morton codes, the coordinates are recovered by extracting every
    third bit:

    .. math::

        x = \sum_{i=0}^{20} \text{bit}_{3i}(\text{morton}) \cdot 2^i

        y = \sum_{i=0}^{20} \text{bit}_{3i+1}(\text{morton}) \cdot 2^i

        z = \sum_{i=0}^{20} \text{bit}_{3i+2}(\text{morton}) \cdot 2^i

    For 2D Morton codes, every second bit is extracted:

    .. math::

        x = \sum_{i=0}^{31} \text{bit}_{2i}(\text{morton}) \cdot 2^i

        y = \sum_{i=0}^{31} \text{bit}_{2i+1}(\text{morton}) \cdot 2^i

    Parameters
    ----------
    codes : Tensor, shape (...)
        Morton codes to decode.
    dimensions : int, default=3
        Number of dimensions (2 or 3).

    Returns
    -------
    Tensor, shape (..., dimensions), dtype=int64
        Integer coordinates.

    Examples
    --------
    3D decoding:

    >>> codes = torch.tensor([0, 1, 2, 3, 7])
    >>> morton_decode(codes, dimensions=3)
    tensor([[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1]])

    2D decoding:

    >>> codes = torch.tensor([0, 1, 2, 3])
    >>> morton_decode(codes, dimensions=2)
    tensor([[0, 0],
            [1, 0],
            [0, 1],
            [1, 1]])

    Batched decoding:

    >>> codes = torch.tensor([[0, 7], [56, 63]])
    >>> morton_decode(codes, dimensions=3)
    tensor([[[0, 0, 0],
             [1, 1, 1]],
            [[2, 2, 2],
             [3, 3, 3]]])

    Round-trip verification:

    >>> coords = torch.tensor([[5, 10, 15], [100, 200, 50]])
    >>> morton_decode(morton_encode(coords), dimensions=3)
    tensor([[  5,  10,  15],
            [100, 200,  50]])

    Notes
    -----
    - This is the inverse of :func:`morton_encode`
    - The dimensions parameter must match the encoding used
    - Invalid Morton codes (not produced by encoding) will decode to
      some coordinates, but the results may not be meaningful

    See Also
    --------
    morton_encode : Encode coordinates to Morton codes
    """
    return torch.ops.torchscience.morton_decode(codes, dimensions)
