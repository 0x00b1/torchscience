"""sRGB to HSV color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_hsv(input: Tensor) -> Tensor:
    r"""Convert sRGB to HSV color space.

    Converts input colors from sRGB to HSV (Hue, Saturation, Value) color space.
    The conversion is differentiable and supports arbitrary batch dimensions.

    Mathematical Definition
    -----------------------
    Given input :math:`(R, G, B)`:

    .. math::
        V &= \max(R, G, B) \\
        S &= \frac{V - \min(R, G, B)}{V} \quad \text{if } V \neq 0 \\
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
        HSV color values where:

        - H (hue): in [0, 2π] radians. 0 = red, 2π/3 = green, 4π/3 = blue.
        - S (saturation): in [0, 1]. 0 = grayscale, 1 = fully saturated.
        - V (value): equals max(R, G, B).

    Examples
    --------
    Convert pure red to HSV:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.graphics.color.srgb_to_hsv(rgb)
    tensor([[0.0000, 1.0000, 1.0000]])

    Convert a batch of colors:

    >>> rgb = torch.tensor([
    ...     [1.0, 0.0, 0.0],  # Red
    ...     [0.0, 1.0, 0.0],  # Green
    ...     [0.0, 0.0, 1.0],  # Blue
    ... ])
    >>> hsv = torchscience.graphics.color.srgb_to_hsv(rgb)
    >>> hsv[:, 0]  # Hue values
    tensor([0.0000, 2.0944, 4.1888])

    Notes
    -----
    - At achromatic points (R = G = B), hue is set to 0 and its gradient is 0.
    - At black (V = 0), saturation is set to 0.
    - Gradients are computed analytically and support backpropagation.

    See Also
    --------
    hsv_to_srgb : Inverse conversion from HSV to sRGB.

    References
    ----------
    .. [1] A. R. Smith, "Color Gamut Transform Pairs", SIGGRAPH 1978.
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_hsv: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_hsv(input)
