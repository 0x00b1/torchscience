"""Inverse Discrete Cosine Transform (IDCT) implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Normalization mode mapping
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
}


def inverse_fourier_cosine_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    type: Literal[1, 2, 3, 4] = 2,
    norm: Optional[Literal["backward", "ortho"]] = None,
) -> Tensor:
    r"""Compute the Inverse Discrete Cosine Transform (IDCT) of a signal.

    This is the inverse of the DCT. IDCT-II is DCT-III (up to scaling),
    and vice versa. DCT-I and DCT-IV are their own inverses.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Must be real-valued.
    n : int, optional
        Signal length. If given, the input will either be padded or
        truncated to this length before computing the transform.
        Default: ``None`` (use input size along ``dim``).
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    type : int, optional
        IDCT type (1, 2, 3, or 4). IDCT-II is computed via DCT-III,
        and IDCT-III via DCT-II.
        Default: ``2``.
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization.
        - ``'ortho'``: Orthonormal normalization.

        Default: ``None`` (equivalent to ``'backward'``).

    Returns
    -------
    Tensor
        The IDCT of the input. Same shape as input with dimension ``dim``
        possibly changed to ``n``.

    Examples
    --------
    Round-trip with forward transform:

    >>> x = torch.randn(32)
    >>> X = torchscience.transform.fourier_cosine_transform(x, norm='ortho')
    >>> x_rec = torchscience.transform.inverse_fourier_cosine_transform(X, norm='ortho')
    >>> torch.allclose(x_rec, x, atol=1e-5)
    True

    See Also
    --------
    fourier_cosine_transform : The forward DCT.
    scipy.fft.idct : SciPy's IDCT implementation.
    """
    if type not in (1, 2, 3, 4):
        raise ValueError(f"type must be 1, 2, 3, or 4, got {type}")

    norm_int = _NORM_MODES.get(norm, 0) if norm is not None else 0

    return torch.ops.torchscience.inverse_fourier_cosine_transform(
        input,
        n if n is not None else -1,
        dim,
        type,
        norm_int,
    )
