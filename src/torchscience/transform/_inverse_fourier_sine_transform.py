"""Inverse Discrete Sine Transform (IDST) implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Normalization mode mapping
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
}


def inverse_fourier_sine_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    type: Literal[1, 2, 3, 4] = 2,
    norm: Optional[Literal["backward", "ortho"]] = None,
) -> Tensor:
    r"""Compute the Inverse Discrete Sine Transform (IDST) of a signal.

    This function computes the inverse of the DST. With orthonormal
    normalization, DST and IDST are exact inverses. With backward
    normalization, IDST includes the appropriate scaling factor.

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
        IDST type (1, 2, 3, or 4).

        - IDST-II uses DST-III with scaling.
        - IDST-III uses DST-II with scaling.
        - IDST-I and IDST-IV are self-inverse.

        Default: ``2``.
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: Includes 1/(2N) scaling.
        - ``'ortho'``: Orthonormal normalization (exact inverse of DST).

        Default: ``None`` (equivalent to ``'backward'``).

    Returns
    -------
    Tensor
        The IDST of the input. Same shape as input with dimension ``dim``
        possibly changed to ``n``.

    Examples
    --------
    Round-trip with ortho normalization:

    >>> x = torch.tensor([1., 2., 3., 4.])
    >>> X = fourier_sine_transform(x, norm='ortho')
    >>> x_rec = inverse_fourier_sine_transform(X, norm='ortho')
    >>> torch.allclose(x, x_rec)
    True

    Round-trip with backward normalization:

    >>> X = fourier_sine_transform(x, norm='backward')
    >>> x_rec = inverse_fourier_sine_transform(X, norm='backward')
    >>> torch.allclose(x, x_rec)
    True

    See Also
    --------
    fourier_sine_transform : The forward DST.
    inverse_fourier_cosine_transform : The inverse DCT.
    """
    if type not in (1, 2, 3, 4):
        raise ValueError(f"type must be 1, 2, 3, or 4, got {type}")

    norm_int = _NORM_MODES.get(norm, 0) if norm is not None else 0

    return torch.ops.torchscience.inverse_fourier_sine_transform(
        input,
        n if n is not None else -1,
        dim,
        type,
        norm_int,
    )
