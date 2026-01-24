"""sRGB to HWB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_hwb(input: Tensor) -> Tensor:
    r"""Convert sRGB to HWB color space.

    Converts input colors from sRGB to HWB (Hue, Whiteness, Blackness) color space.
    HWB is an intuitive color model that describes colors in terms of their
    hue, whiteness (amount of white mixed in), and blackness (amount of black mixed in).
    The conversion is differentiable and supports arbitrary batch dimensions.

    Mathematical Definition
    -----------------------
    Given input :math:`(R, G, B)`:

    .. math::
        H &= \text{(same as HSV hue calculation)} \\
        W &= \min(R, G, B) \\
        B &= 1 - \max(R, G, B)

    where H (hue) is computed using the same formula as in HSV conversion.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. No clamping is applied, so values outside
        [0, 1] are allowed (useful for HDR content).

    Returns
    -------
    Tensor, shape (..., 3)
        HWB color values where:

        - H (hue): in [0, 2*pi] radians. 0 = red, 2*pi/3 = green, 4*pi/3 = blue.
        - W (whiteness): in [0, 1]. Amount of white mixed into the color.
        - B (blackness): in [0, 1]. Amount of black mixed into the color.

    Examples
    --------
    Convert pure red to HWB:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.srgb_to_hwb(rgb)
    tensor([[0.0000, 0.0000, 0.0000]])

    Convert white to HWB:

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> hwb = torchscience.color.srgb_to_hwb(rgb)
    >>> hwb[:, 1]  # Whiteness = 1
    tensor([1.0000])

    Convert black to HWB:

    >>> rgb = torch.tensor([[0.0, 0.0, 0.0]])
    >>> hwb = torchscience.color.srgb_to_hwb(rgb)
    >>> hwb[:, 2]  # Blackness = 1
    tensor([1.0000])

    Notes
    -----
    - At achromatic points (R = G = B), hue is set to 0 and its gradient is 0.
    - W + B = 1 for grayscale colors (white, black, and all grays).
    - Gradients are computed analytically and support backpropagation.

    See Also
    --------
    hwb_to_srgb : Inverse conversion from HWB to sRGB.
    srgb_to_hsv : Convert sRGB to HSV color space.

    References
    ----------
    .. [1] A. Van Wyck, "HWB - A More Intuitive Hue-Based Color Model",
           Journal of Graphics Tools, 1996.
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_hwb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_hwb(input)
