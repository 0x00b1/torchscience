"""HWB to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def hwb_to_srgb(input: Tensor) -> Tensor:
    r"""Convert HWB to sRGB color space.

    Converts input colors from HWB (Hue, Whiteness, Blackness) to sRGB color space.
    HWB is an intuitive color model that describes colors in terms of their
    hue, whiteness (amount of white mixed in), and blackness (amount of black mixed in).
    The conversion is differentiable and supports arbitrary batch dimensions.

    Mathematical Definition
    -----------------------
    Given input :math:`(H, W, B)` where :math:`H \in [0, 2\pi]`:

    If :math:`W + B \geq 1` (achromatic case):

    .. math::
        R = G = B = \frac{W}{W + B}

    Otherwise (chromatic case):

    .. math::
        S &= 1 - \frac{W}{1 - B} \\
        V &= 1 - B \\
        \text{Then HSV}(H, S, V) \rightarrow \text{RGB using standard algorithm}

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        HWB color values where:

        - H (hue): in [0, 2*pi] radians. 0 = red, 2*pi/3 = green, 4*pi/3 = blue.
        - W (whiteness): typically in [0, 1]. Amount of white mixed in.
        - B (blackness): typically in [0, 1]. Amount of black mixed in.

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values.

    Examples
    --------
    Convert HWB red to RGB:

    >>> hwb = torch.tensor([[0.0, 0.0, 0.0]])  # H=0 (red), W=0, B=0
    >>> torchscience.color.hwb_to_srgb(hwb)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert HWB with 50% whiteness:

    >>> import math
    >>> hwb = torch.tensor([[0.0, 0.5, 0.0]])  # Red with 50% white
    >>> torchscience.color.hwb_to_srgb(hwb)
    tensor([[1.0000, 0.5000, 0.5000]])

    Handle achromatic case (W + B >= 1):

    >>> hwb = torch.tensor([[0.0, 0.6, 0.6]])  # W + B > 1
    >>> rgb = torchscience.color.hwb_to_srgb(hwb)
    >>> rgb  # Results in gray
    tensor([[0.5000, 0.5000, 0.5000]])

    Notes
    -----
    - The hue is treated as cyclic, so values outside [0, 2*pi] wrap around.
    - When W + B >= 1, the color is achromatic (grayscale).
    - Gradients are computed analytically and support backpropagation.
    - At hue sector boundaries, gradients may be discontinuous.

    See Also
    --------
    srgb_to_hwb : Inverse conversion from sRGB to HWB.
    hsv_to_srgb : Convert HSV to sRGB color space.

    References
    ----------
    .. [1] A. Van Wyck, "HWB - A More Intuitive Hue-Based Color Model",
           Journal of Graphics Tools, 1996.
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"hwb_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.hwb_to_srgb(input)
