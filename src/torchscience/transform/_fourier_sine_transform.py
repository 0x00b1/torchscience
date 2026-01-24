"""Discrete Sine Transform (DST) implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Normalization mode mapping
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
}


def fourier_sine_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    type: Literal[1, 2, 3, 4] = 2,
    norm: Optional[Literal["backward", "ortho"]] = None,
) -> Tensor:
    r"""Compute the Discrete Sine Transform (DST) of a signal.

    The DST transforms a sequence of N real numbers into another sequence
    of N real numbers. There are four types of DST (I-IV), with DST-II
    being the most commonly used.

    DST-II (default):

    .. math::
        X[k] = 2 \sum_{n=0}^{N-1} x[n] \sin\left(\frac{\pi (2n+1)(k+1)}{2N}\right)

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
        DST type (1, 2, 3, or 4).

        - Type 1: Boundary conditions assume x[-1] = x[N] = 0.
        - Type 2: The "standard" DST, related to DCT-II.
        - Type 3: Inverse of Type 2 (up to scaling).
        - Type 4: Antisymmetric at both endpoints.

        Default: ``2``.
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization (sum without scaling).
        - ``'ortho'``: Orthonormal normalization (makes DST matrix orthogonal).

        Default: ``None`` (equivalent to ``'backward'``).

    Returns
    -------
    Tensor
        The DST of the input. Same shape as input with dimension ``dim``
        possibly changed to ``n``.

    Examples
    --------
    Basic DST-II:

    >>> x = torch.tensor([1., 2., 3., 4.])
    >>> X = fourier_sine_transform(x)
    >>> X.shape
    torch.Size([4])

    Compare with scipy.fft.dst:

    >>> import scipy.fft
    >>> x_np = x.numpy()
    >>> X_scipy = scipy.fft.dst(x_np, type=2)
    >>> torch.allclose(X, torch.from_numpy(X_scipy).float(), atol=1e-5)
    True

    With orthonormal normalization:

    >>> X_ortho = fourier_sine_transform(x, norm='ortho')

    Notes
    -----
    **DST Types:**

    - **Type I (DST-I)**: The transform is its own inverse (up to scaling).
    - **Type II (DST-II)**: Most common DST. Its inverse is DST-III.
    - **Type III (DST-III)**: Inverse of DST-II.
    - **Type IV (DST-IV)**: The transform is its own inverse.

    **Applications:**

    - DST is used in signal processing where odd boundary conditions are natural.
    - DST-I is used in solving differential equations with Dirichlet boundary
      conditions (zero at boundaries).
    - DST is related to Fourier series of odd functions.

    **Implementation:**

    DST-II and DST-III are computed via FFT by constructing an appropriate
    antisymmetric sequence. DST-I and DST-IV use direct matrix multiplication.

    See Also
    --------
    inverse_fourier_sine_transform : The inverse DST.
    fourier_cosine_transform : The related Discrete Cosine Transform.
    scipy.fft.dst : SciPy's DST implementation.
    """
    if type not in (1, 2, 3, 4):
        raise ValueError(f"type must be 1, 2, 3, or 4, got {type}")

    norm_int = _NORM_MODES.get(norm, 0) if norm is not None else 0

    return torch.ops.torchscience.fourier_sine_transform(
        input,
        n if n is not None else -1,
        dim,
        type,
        norm_int,
    )
