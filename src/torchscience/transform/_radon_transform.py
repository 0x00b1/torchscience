"""Radon transform implementation."""

import torch
from torch import Tensor

__all__ = ["radon_transform"]


def radon_transform(
    input: Tensor,
    angles: Tensor,
    *,
    circle: bool = True,
) -> Tensor:
    r"""Compute the Radon transform (sinogram) of a 2D image.

    The Radon transform computes line integrals through a 2D image at
    various angles. For each angle :math:`\theta`, it computes:

    .. math::

        R_\theta(s) = \int_{-\infty}^{\infty} f(s\cos\theta - t\sin\theta,
                                                s\sin\theta + t\cos\theta) \, dt

    where :math:`s` is the perpendicular distance from the origin to the line.

    Parameters
    ----------
    input : Tensor
        2D input image of shape ``[H, W]`` or batched ``[..., H, W]``.
    angles : Tensor
        1D tensor of projection angles in radians.
    circle : bool, optional
        If True, inscribe the image in a circle to avoid corner artifacts.
        Default: True.

    Returns
    -------
    Tensor
        Sinogram of shape ``[..., num_angles, num_bins]`` where ``num_bins``
        is approximately ``ceil(sqrt(H^2 + W^2))``.

    Examples
    --------
    Compute the Radon transform of a simple phantom:

    >>> import torch
    >>> import torchscience.transform as T
    >>> import math
    >>> # Create a simple square phantom
    >>> phantom = torch.zeros(64, 64, dtype=torch.float64)
    >>> phantom[20:44, 20:44] = 1.0
    >>> # Projection angles from 0 to pi
    >>> angles = torch.linspace(0, math.pi, 180, dtype=torch.float64)
    >>> sinogram = T.radon_transform(phantom, angles)

    Notes
    -----
    The Radon transform is the mathematical basis for computed tomography (CT).
    The inverse Radon transform can be computed using filtered back-projection
    or iterative reconstruction algorithms.

    For a square image, the number of detector bins is approximately the
    diagonal length of the image to capture the full projection at all angles.

    When ``circle=True``, pixels outside the inscribed circle are set to zero
    before projection, which helps avoid artifacts from the corners of a
    square image.

    See Also
    --------
    inverse_radon_transform : Filtered back-projection reconstruction.
    """
    return torch.ops.torchscience.radon_transform(input, angles, circle)
