"""Mathematical morphology opening operation."""

from typing import Sequence

from torch import Tensor

from torchscience.morphology._dilation import dilation
from torchscience.morphology._erosion import erosion


def opening(
    input: Tensor,
    structuring_element: Tensor,
    *,
    origin: Sequence[int] | None = None,
    padding_mode: str = "zeros",
) -> Tensor:
    r"""Compute N-dimensional morphological opening.

    Opening is erosion followed by dilation with the same structuring element.
    It removes small bright spots and smooths object boundaries while
    approximately preserving object sizes and shapes.

    Mathematical Definition
    -----------------------
    .. math::
        \gamma_B(f) = \delta_B(\varepsilon_B(f))

    where :math:`\varepsilon_B` is erosion and :math:`\delta_B` is dilation.

    Parameters
    ----------
    input : Tensor, shape (*, spatial...)
        Input tensor. The trailing dimensions are treated as spatial dimensions
        matching the structuring element's dimensions.
    structuring_element : Tensor
        Structuring element. If dtype is bool, flat morphology is performed.
        If dtype is float, grayscale (non-flat) morphology is performed.
    origin : Sequence[int], optional
        Origin (anchor point) of the structuring element. Default is the center.
    padding_mode : str, optional
        How to handle boundaries. One of ``"zeros"``, ``"reflect"``,
        ``"replicate"``, ``"circular"``. Default: ``"zeros"``.

    Returns
    -------
    Tensor
        Opened tensor with the same shape as input.

    Examples
    --------
    Remove small bright features:

    >>> image = torch.rand(64, 64)
    >>> se = torch.ones(3, 3, dtype=torch.bool)
    >>> opened = torchscience.morphology.opening(image, se)

    Notes
    -----
    - Opening is idempotent: ``opening(opening(f, B), B) = opening(f, B)``.
    - Opening is anti-extensive: ``opening(f, B) <= f``.
    - Opening preserves relative ordering: if ``f <= g``, then
      ``opening(f, B) <= opening(g, B)``.

    See Also
    --------
    closing : Dual operation (dilation followed by erosion).
    erosion : First step of opening.
    dilation : Second step of opening.

    References
    ----------
    .. [1] J. Serra, "Image Analysis and Mathematical Morphology", Academic Press, 1982.
    """
    eroded = erosion(
        input,
        structuring_element,
        origin=origin,
        padding_mode=padding_mode,
    )
    return dilation(
        eroded,
        structuring_element,
        origin=origin,
        padding_mode=padding_mode,
    )
