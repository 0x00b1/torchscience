"""Vector refraction using Snell's law."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def refract(direction: Tensor, normal: Tensor, eta: Tensor | float) -> Tensor:
    r"""Refract a direction vector through a surface using Snell's law.

    Computes the refracted ray direction when light passes from one medium
    to another. This is commonly used in ray tracing for rendering transparent
    materials like glass, water, and diamonds.

    Mathematical Definition
    -----------------------
    Given direction :math:`D`, normal :math:`N`, and refractive index ratio
    :math:`\eta = n_1 / n_2`:

    .. math::
        T = \eta D + (\eta \cos\theta_i - \cos\theta_t) N

    where:

    - :math:`\cos\theta_i = -D \cdot N` (incident angle cosine)
    - :math:`\cos\theta_t = \sqrt{1 - \eta^2 \sin^2\theta_i}` (transmitted angle cosine)

    When :math:`\sin^2\theta_t > 1` (total internal reflection), returns a zero vector.

    Parameters
    ----------
    direction : Tensor, shape (..., 3)
        Incident direction vectors pointing toward the surface. Should be
        normalized for physically correct refraction.

    normal : Tensor, shape (..., 3)
        Surface normal vectors pointing away from the surface (into the medium
        the ray is coming from). Should be normalized.

    eta : Tensor or float
        Ratio of refractive indices :math:`n_1 / n_2`, where :math:`n_1` is the
        refractive index of the medium the ray is coming from, and :math:`n_2`
        is the refractive index of the medium the ray is entering. Can be a
        scalar (applied to all rays) or a tensor matching the batch dimensions.

        Common values:
        - Air to glass: eta = 1.0 / 1.5 = 0.667
        - Glass to air: eta = 1.5 / 1.0 = 1.5
        - Air to water: eta = 1.0 / 1.33 = 0.75
        - Air to diamond: eta = 1.0 / 2.42 = 0.41

    Returns
    -------
    Tensor, shape (..., 3)
        Refracted direction vectors. Returns zero vector for rays experiencing
        total internal reflection (TIR).

    Examples
    --------
    Light entering glass from air at normal incidence:

    >>> direction = torch.tensor([0.0, -1.0, 0.0])
    >>> normal = torch.tensor([0.0, 1.0, 0.0])
    >>> eta = 1.0 / 1.5  # air to glass
    >>> refract(direction, normal, eta)
    tensor([0., -1., 0.])

    Light at an angle (30 degrees from normal):

    >>> import math
    >>> theta = math.radians(30)
    >>> direction = torch.tensor([math.sin(theta), -math.cos(theta), 0.0])
    >>> normal = torch.tensor([0.0, 1.0, 0.0])
    >>> refracted = refract(direction, normal, 1.0 / 1.5)

    Total internal reflection (light trying to exit glass at steep angle):

    >>> direction = torch.tensor([0.866, -0.5, 0.0])  # 60 degrees
    >>> normal = torch.tensor([0.0, 1.0, 0.0])
    >>> eta = 1.5  # glass to air
    >>> refract(direction, normal, eta)
    tensor([0., 0., 0.])  # TIR - returns zero

    Batch of rays with different eta values:

    >>> directions = torch.randn(100, 3)
    >>> directions = directions / directions.norm(dim=-1, keepdim=True)
    >>> normals = torch.tensor([[0.0, 1.0, 0.0]]).expand(100, 3)
    >>> etas = torch.rand(100) * 0.5 + 0.5  # random values 0.5-1.0
    >>> refracted = refract(directions, normals, etas)

    Notes
    -----
    - Both direction and normal should be normalized for correct results.
    - The normal should point into the medium the ray is coming from.
    - For rays undergoing total internal reflection, the function returns
      a zero vector. Use :func:`reflect` to compute the reflected ray.
    - This function is differentiable with respect to all inputs.

    See Also
    --------
    reflect : Compute reflected rays off a surface.

    References
    ----------
    .. [1] Pharr, M., Jakob, W., & Humphreys, G. (2016). Physically Based Rendering:
           From Theory to Implementation (3rd ed.). Morgan Kaufmann.
    .. [2] Hecht, E. (2017). Optics (5th ed.). Pearson.
    """
    if direction.shape[-1] != 3:
        raise ValueError(
            f"refract: direction must have last dimension 3, got {direction.shape[-1]}"
        )

    if normal.shape[-1] != 3:
        raise ValueError(
            f"refract: normal must have last dimension 3, got {normal.shape[-1]}"
        )

    # Convert eta to tensor if needed
    if not isinstance(eta, Tensor):
        eta = torch.tensor(eta, dtype=direction.dtype, device=direction.device)

    return torch.ops.torchscience.refract(direction, normal, eta)
