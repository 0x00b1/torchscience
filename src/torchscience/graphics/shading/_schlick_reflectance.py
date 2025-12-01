"""Schlick reflectance approximation implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def schlick_reflectance(
    cosine: Tensor,
    *,
    ior: Union[Tensor, float],
) -> Tensor:
    r"""Compute Schlick approximation for Fresnel reflectance.

    Evaluates the Schlick approximation to the Fresnel equations, which
    describes how light reflects at the interface between two materials.

    Mathematical Definition
    -----------------------
    First, the reflectance at normal incidence is computed:

    .. math::
        r_0 = \left(\frac{1 - \eta}{1 + \eta}\right)^2

    where :math:`\eta` is the index of refraction (ior).

    Then, the reflectance at any angle is:

    .. math::
        R = r_0 + (1 - r_0)(1 - \cos\theta)^5

    where :math:`\cos\theta` is the cosine of the angle between the
    incident direction and the surface normal.

    Parameters
    ----------
    cosine : Tensor, shape (...)
        Cosine of the angle between the incident direction and normal.
        Should be in the range [0, 1] for physically meaningful results.
    ior : Tensor or float, shape (...) or scalar
        Index of refraction of the material. Common values:
        - Air: 1.0
        - Water: 1.33
        - Glass: 1.5
        - Diamond: 2.42

    Returns
    -------
    Tensor, shape (...)
        Fresnel reflectance values in [r0, 1], where r0 is the
        reflectance at normal incidence.

    Examples
    --------
    Glass with ior=1.5 (r0 = 0.04):

    >>> cosine = torch.tensor([1.0, 0.5, 0.0])  # Normal, 60 deg, grazing
    >>> torchscience.graphics.shading.schlick_reflectance(cosine, ior=1.5)
    tensor([0.0400, 0.0700, 1.0000])

    Notes
    -----
    - At normal incidence (cosine=1), reflectance equals r0.
    - At grazing angles (cosine=0), reflectance approaches 1.
    - The function is monotonically decreasing in cosine.
    - **Gradient Support**: Gradients are computed with respect to cosine.
      The ior parameter does not support gradients.

    References
    ----------
    .. [1] C. Schlick, "An Inexpensive BRDF Model for Physically-based
           Rendering", Computer Graphics Forum, 1994.
    """
    # Convert ior to tensor if needed
    if not isinstance(ior, Tensor):
        ior = torch.tensor(ior, device=cosine.device, dtype=cosine.dtype)
    else:
        # Ensure same dtype and device
        if ior.dtype != cosine.dtype:
            ior = ior.to(cosine.dtype)
        if ior.device != cosine.device:
            ior = ior.to(cosine.device)

    # Compute r0 = ((1 - ior) / (1 + ior))^2
    one = torch.ones_like(ior)
    r0 = ((one - ior) / (one + ior)) ** 2

    return torch.ops.torchscience.schlick_reflectance(cosine, r0)
