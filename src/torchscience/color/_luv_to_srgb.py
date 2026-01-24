"""CIELUV to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def luv_to_srgb(input: Tensor) -> Tensor:
    r"""Convert CIELUV to sRGB color space.

    Converts input colors from CIELUV (L*u*v*) to sRGB (with gamma) color space.
    Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    The conversion first transforms LUV to XYZ, then converts to linear RGB
    and applies gamma encoding.

    From L*, compute Y:

    .. math::
        Y = \begin{cases}
            Y_n \cdot \left(\frac{L^* + 16}{116}\right)^3 & \text{if } L^* > 8 \\
            Y_n \cdot \frac{L^*}{903.3} & \text{otherwise}
        \end{cases}

    From u* and v*, recover the chromaticity coordinates:

    .. math::
        u' = \frac{u^*}{13 L^*} + u'_n, \quad v' = \frac{v^*}{13 L^*} + v'_n

    Then compute X and Z:

    .. math::
        X = Y \cdot \frac{9 u'}{4 v'}, \quad Z = Y \cdot \frac{12 - 3u' - 20v'}{4v'}

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        CIELUV color values [L*, u*, v*] where:
        - L* is lightness (typically [0, 100] for SDR)
        - u* is typically in [-100, 100]
        - v* is typically in [-100, 100]

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values. Values may be outside [0, 1] for out-of-gamut colors.

    Examples
    --------
    Convert LUV white to sRGB (should give R=G=B=1):

    >>> luv = torch.tensor([[100.0, 0.0, 0.0]])
    >>> torchscience.color.luv_to_srgb(luv)
    tensor([[1.0000, 1.0000, 1.0000]])

    Convert LUV red back to sRGB:

    >>> luv = torch.tensor([[53.2329, 175.0150, 37.7564]])
    >>> rgb = torchscience.color.luv_to_srgb(luv)
    >>> rgb[:, 0]  # R component (should be close to 1)
    tensor([1.0000])

    See Also
    --------
    srgb_to_luv : Inverse conversion from sRGB to CIELUV.
    xyz_to_srgb : Convert XYZ to sRGB (intermediate space).
    lab_to_srgb : Convert CIELAB to sRGB (similar perceptual space).

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"luv_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.luv_to_srgb(input)
