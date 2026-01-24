"""Oklab to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def oklab_to_srgb(input: Tensor) -> Tensor:
    r"""Convert Oklab to sRGB color space.

    Converts input colors from Oklab, a perceptually uniform color space,
    back to sRGB (with gamma encoding).

    Mathematical Definition
    -----------------------
    The conversion applies inverse transforms:

    1. Apply inverse M2 matrix to get L'M'S':

    .. math::
        \begin{pmatrix} l' \\ m' \\ s' \end{pmatrix} =
        M_2^{-1} \begin{pmatrix} L \\ a \\ b \end{pmatrix}

    2. Cube to get LMS:

    .. math::
        l = (l')^3, \quad m = (m')^3, \quad s = (s')^3

    3. Apply inverse M1 matrix to get linear RGB:

    .. math::
        \begin{pmatrix} R_{linear} \\ G_{linear} \\ B_{linear} \end{pmatrix} =
        M_1^{-1} \begin{pmatrix} l \\ m \\ s \end{pmatrix}

    4. Apply sRGB gamma encoding

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Oklab color values [L, a, b].

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values [R, G, B]. Values may be outside [0, 1] for
        out-of-gamut colors.

    Examples
    --------
    Convert Oklab white to sRGB:

    >>> oklab = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.oklab_to_srgb(oklab)
    tensor([[1.0000, 1.0000, 1.0000]])

    Round-trip conversion:

    >>> rgb = torch.tensor([[0.5, 0.3, 0.7]])
    >>> oklab = torchscience.color.srgb_to_oklab(rgb)
    >>> rgb_back = torchscience.color.oklab_to_srgb(oklab)
    >>> torch.allclose(rgb, rgb_back, atol=1e-5)
    True

    See Also
    --------
    srgb_to_oklab : Forward conversion from sRGB to Oklab.
    lab_to_srgb : Convert CIELAB to sRGB (different from Oklab).

    References
    ----------
    .. [1] Bjorn Ottosson, "A perceptual color space for image processing",
           https://bottosson.github.io/posts/oklab/
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"oklab_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.oklab_to_srgb(input)
