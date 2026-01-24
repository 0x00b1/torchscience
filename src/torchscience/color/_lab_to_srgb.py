"""CIELAB to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def lab_to_srgb(input: Tensor) -> Tensor:
    r"""Convert CIELAB to sRGB color space.

    Converts input colors from CIELAB (L*a*b*) to sRGB (with gamma) color space.
    Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    The conversion first transforms Lab to XYZ, then converts to linear RGB
    and applies gamma encoding:

    .. math::
        f_y = \frac{L^* + 16}{116}

    .. math::
        f_x = \frac{a^*}{500} + f_y

    .. math::
        f_z = f_y - \frac{b^*}{200}

    Then the inverse f function is applied:

    .. math::
        f^{-1}(t) = \begin{cases}
            t^3 & \text{if } t > \delta \\
            3\delta^2 (t - \frac{4}{29}) & \text{otherwise}
        \end{cases}

    with :math:`\delta = 6/29`, to recover XYZ values via
    :math:`X = X_n \cdot f^{-1}(f_x)`, etc., where
    :math:`(X_n, Y_n, Z_n) = (0.95047, 1.0, 1.08883)` is the D65 white point.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        CIELAB color values [L*, a*, b*] where:
        - L* is lightness (typically [0, 100] for SDR)
        - a* is green-red axis (typically [-128, 127])
        - b* is blue-yellow axis (typically [-128, 127])

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values. Values may be outside [0, 1] for out-of-gamut colors.

    Examples
    --------
    Convert Lab white to sRGB (should give R=G=B=1):

    >>> lab = torch.tensor([[100.0, 0.0, 0.0]])
    >>> torchscience.color.lab_to_srgb(lab)
    tensor([[1.0000, 1.0000, 1.0000]])

    Convert pure Lab red back to sRGB:

    >>> lab = torch.tensor([[53.2329, 80.1093, 67.2201]])
    >>> rgb = torchscience.color.lab_to_srgb(lab)
    >>> rgb[:, 0]  # R component (should be close to 1)
    tensor([1.0000])

    See Also
    --------
    srgb_to_lab : Inverse conversion from sRGB to CIELAB.
    xyz_to_srgb : Convert XYZ to sRGB (intermediate space).

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"lab_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.lab_to_srgb(input)
