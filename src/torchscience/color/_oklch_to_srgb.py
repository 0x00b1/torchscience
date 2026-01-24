"""Oklch to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def oklch_to_srgb(input: Tensor) -> Tensor:
    r"""Convert Oklch to sRGB color space.

    Converts input colors from Oklch (cylindrical Oklab) to sRGB (with gamma).
    Oklch is designed by Bjorn Ottosson as a perceptually uniform color space.

    Mathematical Definition
    -----------------------
    The conversion first transforms Oklch to Oklab:

    .. math::
        L = L

    .. math::
        a = C \cdot \cos(h)

    .. math::
        b = C \cdot \sin(h)

    Then converts Oklab to sRGB via the inverse matrix transforms.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Oklch color values [L, C, h] where:
        - L is lightness in [0, 1] for SDR content
        - C is chroma (saturation), non-negative
        - h is hue angle in radians

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values. Values may be outside [0, 1] for out-of-gamut colors.

    Examples
    --------
    Convert Oklch white to sRGB:

    >>> lch = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.oklch_to_srgb(lch)
    tensor([[1.0000, 1.0000, 1.0000]])

    Convert a saturated color:

    >>> import math
    >>> lch = torch.tensor([[0.7, 0.15, math.pi / 6]])  # h = 30 degrees
    >>> rgb = torchscience.color.oklch_to_srgb(lch)

    See Also
    --------
    srgb_to_oklch : Inverse conversion from sRGB to Oklch.
    oklab_to_srgb : Convert Oklab (rectangular form) to sRGB.
    lchab_to_srgb : Convert LCHab (cylindrical CIELAB) to sRGB.

    References
    ----------
    .. [1] Bjorn Ottosson, "A perceptual color space for image processing",
           https://bottosson.github.io/posts/oklab/
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"oklch_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.oklch_to_srgb(input)
