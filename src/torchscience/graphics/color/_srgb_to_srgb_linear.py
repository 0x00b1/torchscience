"""sRGB to linear sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_srgb_linear(input: Tensor) -> Tensor:
    r"""Convert sRGB to linear sRGB color space.

    Converts input colors from sRGB (gamma-encoded) to linear sRGB color space
    using the IEC 61966-2-1 standard transfer function. The conversion is
    differentiable and supports arbitrary tensor shapes.

    Mathematical Definition
    -----------------------
    For each input value :math:`x`:

    .. math::
        f(x) = \begin{cases}
            \frac{x}{12.92} & \text{if } x \leq 0.04045 \\
            \left(\frac{x + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
        \end{cases}

    The backward gradient is:

    .. math::
        \frac{\partial f}{\partial x} = \begin{cases}
            \frac{1}{12.92} & \text{if } x \leq 0.04045 \\
            \frac{2.4}{1.055} \left(\frac{x + 0.055}{1.055}\right)^{1.4} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    input : Tensor
        sRGB color values, typically in the range [0, 1]. Can be any shape,
        e.g., (3,) for a single RGB pixel, (H, W, 3) for an image, or
        (B, H, W, 3) for a batch of images.

    Returns
    -------
    Tensor
        Linear sRGB color values with the same shape as input.

    Examples
    --------
    Convert a single sRGB value:

    >>> srgb = torch.tensor([0.5])
    >>> torchscience.graphics.color.srgb_to_srgb_linear(srgb)
    tensor([0.2140])

    Convert an RGB triplet:

    >>> srgb = torch.tensor([0.2, 0.5, 0.8])
    >>> torchscience.graphics.color.srgb_to_srgb_linear(srgb)
    tensor([0.0331, 0.2140, 0.6038])

    Convert values in the linear region (below threshold 0.04045):

    >>> srgb = torch.tensor([0.01, 0.02, 0.04])
    >>> linear = torchscience.graphics.color.srgb_to_srgb_linear(srgb)
    >>> linear  # linear = srgb / 12.92
    tensor([0.0008, 0.0015, 0.0031])

    Notes
    -----
    - The threshold value 0.04045 is defined by IEC 61966-2-1.
    - Values below the threshold use a linear formula to avoid numerical
      issues near zero.
    - Gradients are computed analytically and support backpropagation.
    - The function is continuous at the threshold point.

    See Also
    --------
    srgb_linear_to_srgb : Inverse conversion from linear sRGB to sRGB.

    References
    ----------
    .. [1] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management - Default
           RGB colour space - sRGB"
    """
    return torch.ops.torchscience.srgb_to_srgb_linear(input)
