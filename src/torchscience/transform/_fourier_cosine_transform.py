"""Discrete Cosine Transform (DCT) implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Normalization mode mapping
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
}


def fourier_cosine_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    type: Literal[1, 2, 3, 4] = 2,
    norm: Optional[Literal["backward", "ortho"]] = None,
) -> Tensor:
    r"""Compute the Discrete Cosine Transform (DCT) of a signal.

    The DCT transforms a sequence of N real numbers into another sequence
    of N real numbers. There are four types of DCT (I-IV), with DCT-II
    being the most commonly used.

    DCT-II (default):

    .. math::
        X[k] = 2 \sum_{n=0}^{N-1} x[n] \cos\left(\frac{\pi k (2n+1)}{2N}\right)

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
        DCT type (1, 2, 3, or 4).

        - Type 1: Boundary conditions assume x[-1] = x[0], x[N] = x[N-1].
        - Type 2: The "standard" DCT used in JPEG, MP3.
        - Type 3: Inverse of Type 2 (up to scaling).
        - Type 4: Symmetric at both endpoints.

        Default: ``2``.
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization (sum without scaling).
        - ``'ortho'``: Orthonormal normalization (makes DCT matrix orthogonal).

        Default: ``None`` (equivalent to ``'backward'``).

    Returns
    -------
    Tensor
        The DCT of the input. Same shape as input with dimension ``dim``
        possibly changed to ``n``.

    Examples
    --------
    Basic DCT-II:

    >>> x = torch.tensor([1., 2., 3., 4.])
    >>> X = fourier_cosine_transform(x)
    >>> X.shape
    torch.Size([4])

    Compare with scipy.fft.dct:

    >>> import scipy.fft
    >>> x_np = x.numpy()
    >>> X_scipy = scipy.fft.dct(x_np, type=2)
    >>> torch.allclose(X, torch.from_numpy(X_scipy).float(), atol=1e-5)
    True

    With orthonormal normalization:

    >>> X_ortho = fourier_cosine_transform(x, norm='ortho')

    Notes
    -----
    **DCT Types:**

    - **Type I (DCT-I)**: Requires N >= 2. The transform is its own inverse.
    - **Type II (DCT-II)**: Most common DCT, used in JPEG, MP3. Its inverse
      is DCT-III.
    - **Type III (DCT-III)**: Inverse of DCT-II. Sometimes called IDCT.
    - **Type IV (DCT-IV)**: The transform is its own inverse.

    **Applications:**

    - DCT-II is used in lossy compression (JPEG, MP3, MPEG).
    - DCT is related to PCA and is optimal for certain signal classes.
    - DCT has better energy compaction than DFT for many signal types.

    **Implementation:**

    Computed via FFT by constructing an appropriate symmetric sequence.

    See Also
    --------
    inverse_fourier_cosine_transform : The inverse DCT.
    scipy.fft.dct : SciPy's DCT implementation.
    """
    if type not in (1, 2, 3, 4):
        raise ValueError(f"type must be 1, 2, 3, or 4, got {type}")

    norm_int = _NORM_MODES.get(norm, 0) if norm is not None else 0

    return torch.ops.torchscience.fourier_cosine_transform(
        input,
        n if n is not None else -1,
        dim,
        type,
        norm_int,
    )
