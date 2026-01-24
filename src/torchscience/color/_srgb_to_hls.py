"""sRGB to HLS color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_hls(input: Tensor) -> Tensor:
    r"""Convert sRGB to HLS color space.

    Converts input colors from sRGB to HLS (Hue, Lightness, Saturation) color space.
    The conversion is differentiable and supports arbitrary batch dimensions.

    Mathematical Definition
    -----------------------
    Given input :math:`(R, G, B)`:

    .. math::
        L &= \frac{\max(R, G, B) + \min(R, G, B)}{2} \\
        S &= \begin{cases}
            \frac{\Delta}{\max + \min} & \text{if } L \leq 0.5 \\
            \frac{\Delta}{2 - \max - \min} & \text{if } L > 0.5
        \end{cases} \\
        H &= \frac{\pi}{3} \times \begin{cases}
            \frac{G - B}{\Delta} \mod 6 & \text{if } R = \max \\
            \frac{B - R}{\Delta} + 2 & \text{if } G = \max \\
            \frac{R - G}{\Delta} + 4 & \text{if } B = \max
        \end{cases}

    where :math:`\Delta = \max(R, G, B) - \min(R, G, B)`.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. No clamping is applied, so values outside
        [0, 1] are allowed (useful for HDR content).

    Returns
    -------
    Tensor, shape (..., 3)
        HLS color values where:

        - H (hue): in [0, 2pi] radians. 0 = red, 2pi/3 = green, 4pi/3 = blue.
        - L (lightness): in [0, 1]. 0 = black, 1 = white, 0.5 = pure color.
        - S (saturation): in [0, 1]. 0 = grayscale, 1 = fully saturated.

    Examples
    --------
    Convert pure red to HLS:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.srgb_to_hls(rgb)
    tensor([[0.0000, 0.5000, 1.0000]])

    Convert a batch of colors:

    >>> rgb = torch.tensor([
    ...     [1.0, 0.0, 0.0],  # Red
    ...     [0.0, 1.0, 0.0],  # Green
    ...     [0.0, 0.0, 1.0],  # Blue
    ... ])
    >>> hls = torchscience.color.srgb_to_hls(rgb)
    >>> hls[:, 0]  # Hue values
    tensor([0.0000, 2.0944, 4.1888])

    Notes
    -----
    - At achromatic points (R = G = B), hue is set to 0 and its gradient is 0.
    - HLS differs from HSV: lightness is the average of max and min, not just max.
    - Pure colors have L = 0.5, while V = 1 in HSV.
    - Gradients are computed analytically and support backpropagation.

    See Also
    --------
    hls_to_srgb : Inverse conversion from HLS to sRGB.
    srgb_to_hsv : Convert sRGB to HSV color space.

    References
    ----------
    .. [1] A. R. Smith, "Color Gamut Transform Pairs", SIGGRAPH 1978.
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_hls: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_hls(input)
