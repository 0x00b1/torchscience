"""sRGB to Oklch color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_oklch(input: Tensor) -> Tensor:
    r"""Convert sRGB to Oklch color space.

    Converts input colors from sRGB (with gamma) to Oklch, the cylindrical
    representation of Oklab. Oklch is designed by Bjorn Ottosson as a
    perceptually uniform color space with intuitive hue and chroma controls.

    Mathematical Definition
    -----------------------
    Oklch is Oklab expressed in cylindrical coordinates:

    .. math::
        L = L \text{ (same as Oklab)}

    .. math::
        C = \sqrt{a^2 + b^2}

    .. math::
        h = \arctan2(b, a)

    where :math:`C` is the chroma (colorfulness) and :math:`h` is the hue angle
    in radians.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        Oklch color values [L, C, h] where:
        - L is lightness in [0, 1] for SDR content
        - C is chroma (saturation), non-negative
        - h is hue angle in radians [-pi, pi]

    Examples
    --------
    Convert white to Oklch (should give L=1, C=0, h=0):

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_oklch(rgb)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert pure red:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> lch = torchscience.color.srgb_to_oklch(rgb)
    >>> lch[:, 0]  # L component
    tensor([0.6280])

    See Also
    --------
    oklch_to_srgb : Inverse conversion from Oklch to sRGB.
    srgb_to_oklab : Convert sRGB to Oklab (rectangular form).
    srgb_to_lchab : Convert sRGB to LCHab (cylindrical CIELAB).

    References
    ----------
    .. [1] Bjorn Ottosson, "A perceptual color space for image processing",
           https://bottosson.github.io/posts/oklab/
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_oklch: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_oklch(input)
