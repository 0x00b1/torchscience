"""Mathematical morphology dilation operation."""

from typing import Sequence

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def dilation(
    input: Tensor,
    structuring_element: Tensor,
    *,
    origin: Sequence[int] | None = None,
    padding_mode: str = "zeros",
) -> Tensor:
    r"""Compute N-dimensional morphological dilation.

    Dilation is one of the fundamental operations in mathematical morphology.
    It expands bright regions (or shrinks dark regions) in an image based on
    the structuring element.

    Mathematical Definition
    -----------------------
    For flat (binary) morphology:

    .. math::
        \delta_B(f)(x) = \max_{b \in B} f(x - b)

    For grayscale (non-flat) morphology:

    .. math::
        \delta_g(f)(x) = \max_{b \in \text{support}(g)} [f(x - b) + g(b)]

    where :math:`B` is the structuring element, :math:`f` is the input,
    and :math:`g` is a grayscale structuring element.

    Parameters
    ----------
    input : Tensor, shape (*, spatial...)
        Input tensor. The trailing dimensions are treated as spatial dimensions
        matching the structuring element's dimensions.
    structuring_element : Tensor
        Structuring element. If dtype is bool, flat morphology is performed
        (values indicate membership in the SE). If dtype is float, grayscale
        (non-flat) morphology is performed (values are added).
    origin : Sequence[int], optional
        Origin (anchor point) of the structuring element. Default is the center
        ``[se.shape[d] // 2 for d in range(se.ndim)]``.
    padding_mode : str, optional
        How to handle boundaries. One of:

        - ``"zeros"``: Pad with -infinity (default for dilation).
        - ``"reflect"``: Reflect at boundaries.
        - ``"replicate"``: Replicate edge values.
        - ``"circular"``: Wrap around (periodic boundary).

        Default: ``"zeros"``.

    Returns
    -------
    Tensor
        Dilated tensor with the same shape as input.

    Examples
    --------
    Binary dilation with a 3x3 square structuring element:

    >>> image = torch.rand(64, 64)
    >>> se = torch.ones(3, 3, dtype=torch.bool)
    >>> dilated = torchscience.morphology.dilation(image, se)

    Batched dilation:

    >>> batch = torch.rand(8, 64, 64)
    >>> se = torch.ones(3, 3, dtype=torch.bool)
    >>> dilated = torchscience.morphology.dilation(batch, se)

    Grayscale (non-flat) dilation:

    >>> image = torch.rand(64, 64)
    >>> se = torch.tensor([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]])
    >>> dilated = torchscience.morphology.dilation(image, se)

    Notes
    -----
    - Gradients are computed with respect to the input tensor using the
      subgradient of the max operation (indicator at argmax position).
    - No gradients flow through the structuring element.
    - For binary images, use dtype=bool for the structuring element.
    - Dilation is the dual of erosion: ``dilation(f, B) = -erosion(-f, B)``.

    See Also
    --------
    erosion : Dual operation (minimum over neighborhood).
    opening : Erosion followed by dilation.
    closing : Dilation followed by erosion.

    References
    ----------
    .. [1] J. Serra, "Image Analysis and Mathematical Morphology", Academic Press, 1982.
    .. [2] P. Soille, "Morphological Image Analysis", Springer, 2004.
    """
    if input.dim() < structuring_element.dim():
        raise ValueError(
            f"dilation: input must have at least as many dimensions as "
            f"structuring_element ({input.dim()} < {structuring_element.dim()})"
        )

    # Convert padding mode string to integer
    padding_modes = {"zeros": 0, "reflect": 1, "replicate": 2, "circular": 3}
    if padding_mode not in padding_modes:
        raise ValueError(
            f"dilation: padding_mode must be one of {list(padding_modes.keys())}, "
            f"got '{padding_mode}'"
        )
    padding_mode_int = padding_modes[padding_mode]

    # Convert origin to list if provided
    origin_list = list(origin) if origin is not None else None

    return torch.ops.torchscience.dilation(
        input, structuring_element, origin_list, padding_mode_int
    )
