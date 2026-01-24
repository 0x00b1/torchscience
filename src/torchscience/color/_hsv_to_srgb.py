"""HSV to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def hsv_to_srgb(input: Tensor) -> Tensor:
    r"""Convert HSV to sRGB color space.

    Converts input colors from HSV (Hue, Saturation, Value) to sRGB color space.
    The conversion is differentiable and supports arbitrary batch dimensions.

    Mathematical Definition
    -----------------------
    Given input :math:`(H, S, V)` where :math:`H \in [0, 2\pi]`:

    .. math::
        C &= V \times S \\
        H' &= H \times \frac{3}{\pi} \\
        X &= C \times (1 - |H' \mod 2 - 1|) \\
        m &= V - C

    Then :math:`(R, G, B) = m + (R_1, G_1, B_1)` where :math:`(R_1, G_1, B_1)`
    depends on which sector of the color wheel :math:`H'` falls into.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        HSV color values where:

        - H (hue): in [0, 2π] radians. 0 = red, 2π/3 = green, 4π/3 = blue.
        - S (saturation): typically in [0, 1].
        - V (value): typically in [0, 1].

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values.

    Examples
    --------
    Convert HSV red to RGB:

    >>> import math
    >>> hsv = torch.tensor([[0.0, 1.0, 1.0]])  # H=0 (red), S=1, V=1
    >>> torchscience.color.hsv_to_srgb(hsv)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert HSV green to RGB:

    >>> hsv = torch.tensor([[2 * math.pi / 3, 1.0, 1.0]])  # H=2π/3 (green)
    >>> torchscience.color.hsv_to_srgb(hsv)
    tensor([[0.0000, 1.0000, 0.0000]])

    Notes
    -----
    - The hue is treated as cyclic, so values outside [0, 2π] wrap around.
    - Gradients are computed analytically and support backpropagation.
    - At hue sector boundaries, gradients may be discontinuous.

    See Also
    --------
    srgb_to_hsv : Inverse conversion from sRGB to HSV.

    References
    ----------
    .. [1] A. R. Smith, "Color Gamut Transform Pairs", SIGGRAPH 1978.
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"hsv_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.hsv_to_srgb(input)
