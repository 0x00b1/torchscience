"""Inverse Hilbert transform implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Padding mode mapping
_PADDING_MODES = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}


def inverse_hilbert_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    padding_mode: Literal[
        "constant", "reflect", "replicate", "circular"
    ] = "constant",
    padding_value: float = 0.0,
    window: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the inverse Hilbert transform of a signal.

    The inverse Hilbert transform recovers the original signal from its
    Hilbert transform. It is defined as:

    .. math::
        \mathcal{H}^{-1}[f] = -\mathcal{H}[f]

    This follows from the property that :math:`\mathcal{H}[\mathcal{H}[f]] = -f`.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Can be real or complex.
    n : int, optional
        Signal length. If given, the input will either be padded or
        truncated to this length before computing the transform.
        Default: ``None`` (use input size along ``dim``).
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    padding_mode : str, optional
        Padding mode when ``n`` is larger than input size. One of:

        - ``'constant'``: Pad with ``padding_value`` (default 0).
        - ``'reflect'``: Reflect the signal at boundaries.
        - ``'replicate'``: Replicate edge values.
        - ``'circular'``: Wrap around (periodic extension).

        Default: ``'constant'``.
    padding_value : float, optional
        Fill value for ``'constant'`` padding mode. Ignored for other modes.
        Default: ``0.0``.
    window : Tensor, optional
        Window function to apply before the transform. Must be 1-D with size
        matching the (possibly padded) signal length along ``dim``.
        Default: ``None`` (no windowing).

    Returns
    -------
    Tensor
        The inverse Hilbert transform of the input.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.

    Examples
    --------
    Verify that inverse undoes forward transform:

    >>> x = torch.randn(100)
    >>> h = torchscience.transform.hilbert_transform(x)
    >>> x_recovered = torchscience.transform.inverse_hilbert_transform(h)
    >>> torch.allclose(x, x_recovered, atol=1e-5)
    True

    Transform along a specific dimension:

    >>> x = torch.randn(3, 100)
    >>> h = inverse_hilbert_transform(x, dim=1)
    >>> h.shape
    torch.Size([3, 100])

    Notes
    -----
    **Mathematical Properties:**

    - :math:`\mathcal{H}^{-1}[\mathcal{H}[f]] = f`
    - :math:`\mathcal{H}[\mathcal{H}^{-1}[f]] = f`
    - :math:`\mathcal{H}^{-1}[f] = -\mathcal{H}[f]`

    **Complex Input Behavior:**

    For complex inputs, the transform is applied linearly to both components:
    :math:`\mathcal{H}^{-1}[a + ib] = \mathcal{H}^{-1}[a] + i\mathcal{H}^{-1}[b]`.

    **Implementation:**

    Uses the same FFT-based approach as the forward transform, but with
    the negated frequency response :math:`h^{-1}[k] = i \cdot \text{sign}(\text{freq}[k])`.

    **Gradient Computation:**

    Gradients are computed analytically. The adjoint of the inverse Hilbert
    transform is the forward Hilbert transform: :math:`(\mathcal{H}^{-1})^T = \mathcal{H}`.
    Therefore:

    .. math::
        \frac{\partial L}{\partial x} = \mathcal{H}\left[\frac{\partial L}{\partial y}\right]

    where :math:`y = \mathcal{H}^{-1}[x]`. Second-order gradients are also supported.

    References
    ----------
    .. [1] F.W. King, "Hilbert Transforms," Cambridge University Press, 2009.

    See Also
    --------
    hilbert_transform : The forward Hilbert transform.
    """
    if padding_mode not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {list(_PADDING_MODES.keys())}, "
            f"got '{padding_mode}'"
        )

    return torch.ops.torchscience.inverse_hilbert_transform(
        input,
        n if n is not None else -1,
        dim,
        _PADDING_MODES[padding_mode],
        padding_value,
        window,
    )
