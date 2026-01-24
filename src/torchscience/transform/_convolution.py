"""FFT-based convolution implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Mode mapping
_MODE_MAP = {
    "full": 0,
    "same": 1,
    "valid": 2,
}


def convolution(
    input: Tensor,
    kernel: Tensor,
    *,
    dim: int = -1,
    mode: Literal["full", "same", "valid"] = "full",
) -> Tensor:
    r"""Compute the convolution of input with kernel along a dimension.

    This function computes the linear convolution using the FFT for efficiency.
    The operation is:

    .. math::
        (f * g)[n] = \sum_{m} f[m] \cdot g[n - m]

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.
    kernel : Tensor
        Convolution kernel. Can be 1-D (applied to the specified dimension)
        or the same shape as input.
    dim : int, optional
        The dimension along which to compute the convolution.
        Default: ``-1`` (last dimension).
    mode : str, optional
        Output size mode:

        - ``'full'``: Return the full convolution (size: N + M - 1).
        - ``'same'``: Return output the same size as input (size: N).
        - ``'valid'``: Return only valid values without padding (size: N - M + 1).

        Default: ``'full'``.

    Returns
    -------
    Tensor
        The convolution of input with kernel.

    Examples
    --------
    Basic convolution:

    >>> x = torch.tensor([1., 2., 3., 4., 5.])
    >>> h = torch.tensor([1., 0., -1.])
    >>> y = convolution(x, h)
    >>> y
    tensor([ 1.,  2.,  2.,  2.,  2., -4., -5.])

    Same mode (preserves input size):

    >>> y_same = convolution(x, h, mode='same')
    >>> y_same.shape
    torch.Size([5])

    Valid mode (no padding):

    >>> y_valid = convolution(x, h, mode='valid')
    >>> y_valid.shape
    torch.Size([3])

    Batched convolution:

    >>> x = torch.randn(3, 5, 100)  # 3x5 batch of 100-element signals
    >>> h = torch.randn(10)  # 10-element kernel
    >>> y = convolution(x, h, dim=-1)
    >>> y.shape
    torch.Size([3, 5, 109])  # full mode: 100 + 10 - 1

    Notes
    -----
    **Implementation:**

    The convolution is computed using the FFT:

    1. Zero-pad input and kernel to length N + M - 1
    2. Compute FFT of both
    3. Multiply in frequency domain
    4. Inverse FFT

    This has complexity O((N+M) log(N+M)) vs O(NM) for direct computation,
    making it efficient for large signals.

    **Relation to other operations:**

    - Convolution: :math:`(f * g)[n] = \sum_m f[m] g[n-m]`
    - Correlation: :math:`(f \star g)[n] = \sum_m f[m] g[n+m] = (f * g^*)[n]`
      where :math:`g^*` is the time-reversed conjugate.

    See Also
    --------
    scipy.signal.convolve : SciPy's convolution function.
    torch.nn.functional.conv1d : PyTorch's neural network convolution.
    """
    if mode not in _MODE_MAP:
        raise ValueError(
            f"mode must be 'full', 'same', or 'valid', got {mode}"
        )

    mode_int = _MODE_MAP[mode]

    return torch.ops.torchscience.convolution(
        input,
        kernel,
        dim,
        mode_int,
    )
