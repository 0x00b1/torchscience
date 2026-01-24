"""HLS to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def hls_to_srgb(input: Tensor) -> Tensor:
    r"""Convert HLS to sRGB color space.

    Converts input colors from HLS (Hue, Lightness, Saturation) to sRGB color space.
    The conversion is differentiable and supports arbitrary batch dimensions.

    Mathematical Definition
    -----------------------
    Given input :math:`(H, L, S)` where :math:`H \in [0, 2\pi]`:

    If :math:`S = 0`, then :math:`R = G = B = L` (achromatic).

    Otherwise:

    .. math::
        q &= \begin{cases}
            L \times (1 + S) & \text{if } L < 0.5 \\
            L + S - L \times S & \text{if } L \geq 0.5
        \end{cases} \\
        p &= 2L - q

    Then each RGB component is computed using a helper function applied to
    different hue offsets.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        HLS color values where:

        - H (hue): in [0, 2pi] radians. 0 = red, 2pi/3 = green, 4pi/3 = blue.
        - L (lightness): typically in [0, 1]. 0 = black, 1 = white.
        - S (saturation): typically in [0, 1].

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values.

    Examples
    --------
    Convert HLS red to RGB:

    >>> hls = torch.tensor([[0.0, 0.5, 1.0]])  # H=0 (red), L=0.5, S=1
    >>> torchscience.color.hls_to_srgb(hls)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert HLS green to RGB:

    >>> import math
    >>> hls = torch.tensor([[2 * math.pi / 3, 0.5, 1.0]])  # H=2pi/3 (green)
    >>> torchscience.color.hls_to_srgb(hls)
    tensor([[0.0000, 1.0000, 0.0000]])

    Notes
    -----
    - The hue is treated as cyclic, so values outside [0, 2pi] wrap around.
    - Gradients are computed analytically and support backpropagation.
    - At hue sector boundaries, gradients may be discontinuous.

    See Also
    --------
    srgb_to_hls : Inverse conversion from sRGB to HLS.
    hsv_to_srgb : Convert HSV to sRGB color space.

    References
    ----------
    .. [1] A. R. Smith, "Color Gamut Transform Pairs", SIGGRAPH 1978.
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"hls_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.hls_to_srgb(input)
