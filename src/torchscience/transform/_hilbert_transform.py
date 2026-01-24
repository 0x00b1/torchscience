"""Hilbert transform implementation."""

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


def hilbert_transform(
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
    r"""Compute the Hilbert transform of a signal along a specified dimension.

    The Hilbert transform is defined as:

    .. math::
        \mathcal{H}[f](x) = \frac{1}{\pi} \text{PV} \int_{-\infty}^{\infty}
        \frac{f(t)}{t - x} \, dt

    where PV denotes the Cauchy principal value.

    For discrete signals, this is computed efficiently using the FFT by
    multiplying the frequency spectrum by :math:`-i \cdot \text{sign}(\omega)`.

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
        Use window functions from ``torch`` (e.g., ``torch.hann_window``) or
        ``torchscience.window_function``.
        Default: ``None`` (no windowing).

    Returns
    -------
    Tensor
        The Hilbert transform of the input.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.
        If input is real, output is real (the imaginary part is discarded).

    Examples
    --------
    Basic usage with a sine wave:

    >>> t = torch.linspace(0, 2 * torch.pi, 100)
    >>> x = torch.sin(t)  # sin(t)
    >>> h = hilbert_transform(x)
    >>> # H[sin(t)] ~ -cos(t) for positive frequencies
    >>> torch.allclose(h, -torch.cos(t), atol=0.1)
    True

    Transform along a specific dimension:

    >>> x = torch.randn(3, 100)
    >>> h = hilbert_transform(x, dim=1)
    >>> h.shape
    torch.Size([3, 100])

    With reflection padding to reduce edge effects:

    >>> x = torch.randn(64)
    >>> h = hilbert_transform(x, n=128, padding_mode='reflect')
    >>> h.shape
    torch.Size([128])

    With a window function:

    >>> x = torch.randn(100)
    >>> window = torch.hann_window(100)
    >>> h = hilbert_transform(x, window=window)

    Notes
    -----
    **Mathematical Properties:**

    - :math:`\mathcal{H}[\sin(\omega t)] = -\cos(\omega t)` (for positive :math:`\omega`)
    - :math:`\mathcal{H}[\cos(\omega t)] = \sin(\omega t)` (for positive :math:`\omega`)
    - :math:`\mathcal{H}[\mathcal{H}[f]] = -f` (involutory up to sign)
    - Energy preservation: :math:`\int |H[f]|^2 = \int |f|^2`
    - Linearity: :math:`\mathcal{H}[\alpha f + \beta g] = \alpha\mathcal{H}[f] + \beta\mathcal{H}[g]`

    **Padding Modes:**

    - ``'constant'``: Zero-padding (default). Simple but can introduce
      discontinuities at boundaries.
    - ``'reflect'``: Reduces edge effects by reflecting the signal. Good for
      non-periodic signals.
    - ``'replicate'``: Extends the signal with edge values. Useful for smooth
      signals.
    - ``'circular'``: Wraps around, assuming periodicity. Best for truly
      periodic signals.

    **Windowing:**

    Applying a window function before the transform can reduce spectral
    leakage and edge effects. Common windows include:

    - ``torch.hann_window``: Good general-purpose window
    - ``torch.hamming_window``: Similar to Hann, slightly different shape
    - ``torch.blackman_window``: Better sidelobe suppression

    **Complex Input Behavior:**

    For complex inputs, the transform is applied linearly to both components:
    :math:`\mathcal{H}[a + ib] = \mathcal{H}[a] + i\mathcal{H}[b]`. This
    preserves conjugate symmetry: :math:`\mathcal{H}[\overline{f}] = \overline{\mathcal{H}[f]}`.

    **Analytic Signal:**

    The analytic signal is defined as :math:`z(t) = f(t) + i\mathcal{H}[f](t)`.
    It can be computed as:

    >>> analytic = x + 1j * hilbert_transform(x)

    **Implementation:**

    Uses FFT-based computation:

    1. Apply padding if ``n`` > input size (using specified mode)
    2. Apply window function if provided
    3. Compute FFT of input
    4. Multiply by frequency response :math:`h[k] = -i \cdot \text{sign}(\text{freq}[k])`
    5. Compute inverse FFT

    **Gradient Computation:**

    Gradients are computed analytically using the property that the Hilbert
    transform is anti-self-adjoint: :math:`\mathcal{H}^T = -\mathcal{H}`.
    Therefore, for a loss :math:`L`:

    .. math::
        \frac{\partial L}{\partial x} = -\mathcal{H}\left[\frac{\partial L}{\partial y}\right]

    where :math:`y = \mathcal{H}[x]`. Second-order gradients are also supported.

    Warnings
    --------
    - Edge effects: The discrete Hilbert transform assumes periodic boundary
      conditions. For non-periodic signals, use ``padding_mode='reflect'``
      or ``padding_mode='replicate'`` with ``n > input.size(dim)``.

    - The transform is not well-defined for DC components (frequency = 0).

    - When using windowing, the window must match the padded signal length,
      not the original input length.

    References
    ----------
    .. [1] F.W. King, "Hilbert Transforms," Cambridge University Press, 2009.

    .. [2] S.L. Hahn, "Hilbert Transforms in Signal Processing,"
           Artech House, 1996.

    See Also
    --------
    inverse_hilbert_transform : The inverse Hilbert transform.
    scipy.signal.hilbert : SciPy's Hilbert transform (returns analytic signal).
    """
    if padding_mode not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {list(_PADDING_MODES.keys())}, "
            f"got '{padding_mode}'"
        )

    return torch.ops.torchscience.hilbert_transform(
        input,
        n if n is not None else -1,
        dim,
        _PADDING_MODES[padding_mode],
        padding_value,
        window,
    )
