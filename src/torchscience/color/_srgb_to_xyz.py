"""sRGB to XYZ color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_xyz(input: Tensor) -> Tensor:
    r"""Convert sRGB to CIE XYZ color space.

    Converts input colors from sRGB (with gamma) to CIE 1931 XYZ color space.
    Uses D65 illuminant and sRGB primaries per IEC 61966-2-1.

    Mathematical Definition
    -----------------------
    The conversion first linearizes sRGB:

    .. math::
        C_{\text{linear}} = \begin{cases}
            C / 12.92 & \text{if } C \leq 0.04045 \\
            \left(\frac{C + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
        \end{cases}

    Then applies the RGB to XYZ matrix:

    .. math::
        \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} =
        \begin{bmatrix}
            0.4124564 & 0.3575761 & 0.1804375 \\
            0.2126729 & 0.7151522 & 0.0721750 \\
            0.0193339 & 0.1191920 & 0.9503041
        \end{bmatrix}
        \begin{bmatrix} R \\ G \\ B \end{bmatrix}

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        CIE XYZ color values where Y is luminance in [0, 1] for SDR content.

    Examples
    --------
    Convert D65 white to XYZ:

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_xyz(rgb)
    tensor([[0.9505, 1.0000, 1.0890]])

    See Also
    --------
    xyz_to_srgb : Inverse conversion from XYZ to sRGB.
    srgb_to_lab : Convert sRGB to CIELAB (goes through XYZ internally).

    References
    ----------
    .. [1] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_xyz: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_xyz(input)
