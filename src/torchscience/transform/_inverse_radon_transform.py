"""Inverse Radon transform (filtered back-projection) implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_radon_transform"]

# Filter type mapping
_FILTER_TYPE_MAP = {
    "ramp": 0,
    "shepp-logan": 1,
    "cosine": 2,
    "hamming": 3,
    "hann": 4,
}


def inverse_radon_transform(
    input: Tensor,
    angles: Tensor,
    *,
    circle: bool = True,
    output_size: int | None = None,
    filter_type: Literal[
        "ramp", "shepp-logan", "cosine", "hamming", "hann"
    ] = "ramp",
) -> Tensor:
    r"""Compute the inverse Radon transform using filtered back-projection.

    The inverse Radon transform reconstructs a 2D image from its sinogram
    (projection data). This implementation uses the filtered back-projection
    (FBP) algorithm:

    1. Apply a ramp filter (or windowed ramp filter) to each projection
       in the frequency domain
    2. Back-project the filtered projections onto the image plane

    .. math::

        f(x, y) \approx \int_0^\pi g_\theta(x\cos\theta + y\sin\theta) \, d\theta

    where :math:`g_\theta` is the filtered projection at angle :math:`\theta`.

    Parameters
    ----------
    input : Tensor
        Sinogram of shape ``[num_angles, num_bins]`` or batched
        ``[..., num_angles, num_bins]``.
    angles : Tensor
        1D tensor of projection angles in radians corresponding to each row
        of the sinogram. Must have length equal to ``num_angles``.
    circle : bool, optional
        If True, reconstruct only the inscribed circle (set pixels outside
        the circle to zero). This matches the ``circle=True`` mode of
        :func:`radon_transform`. Default: ``True``.
    output_size : int, optional
        Size of the output image (produces a square ``output_size x output_size``
        image). If None, the size is automatically computed from the sinogram
        as approximately ``num_bins / sqrt(2)``. Default: ``None``.
    filter_type : str, optional
        Type of filter to use. Options:

        - ``'ramp'``: Standard ramp filter (default). Has the highest spatial
          resolution but amplifies high-frequency noise.
        - ``'shepp-logan'``: Ramp filter multiplied by a sinc window. Good
          balance between resolution and noise.
        - ``'cosine'``: Ramp filter with cosine window. Smoother results.
        - ``'hamming'``: Ramp filter with Hamming window. Reduces ringing.
        - ``'hann'``: Ramp filter with Hann window. Smoothest results.

        Default: ``'ramp'``.

    Returns
    -------
    Tensor
        Reconstructed image of shape ``[..., H, W]`` where ``H = W = output_size``
        (or automatically determined if ``output_size`` is None).

    Examples
    --------
    Round-trip reconstruction:

    >>> import torch
    >>> import torchscience.transform as T
    >>> import math
    >>> # Create a simple phantom
    >>> phantom = torch.zeros(64, 64, dtype=torch.float64)
    >>> phantom[20:44, 20:44] = 1.0
    >>> # Forward Radon transform
    >>> angles = torch.linspace(0, math.pi, 180, dtype=torch.float64)
    >>> sinogram = T.radon_transform(phantom, angles)
    >>> # Inverse Radon transform
    >>> reconstructed = T.inverse_radon_transform(sinogram, angles)
    >>> reconstructed.shape
    torch.Size([64, 64])

    With different filters:

    >>> # Smoother reconstruction with Hamming filter
    >>> recon_smooth = T.inverse_radon_transform(
    ...     sinogram, angles, filter_type="hamming"
    ... )

    Batched reconstruction:

    >>> sinograms = torch.randn(5, 180, 91, dtype=torch.float64)
    >>> angles = torch.linspace(0, math.pi, 180, dtype=torch.float64)
    >>> reconstructed = T.inverse_radon_transform(sinograms, angles)
    >>> reconstructed.shape
    torch.Size([5, 64, 64])

    Notes
    -----
    **Filtered Back-Projection Algorithm:**

    The FBP algorithm is the standard analytical method for CT reconstruction.
    It works by:

    1. Filtering each projection with a ramp filter :math:`|\\omega|` in the
       frequency domain. This compensates for the 1/r blurring inherent in
       back-projection.

    2. Back-projecting the filtered projections: for each pixel, find the
       corresponding value in each filtered projection and sum them.

    **Filter Selection:**

    The ramp filter amplifies high frequencies, which can amplify noise. The
    windowed filters (Shepp-Logan, cosine, Hamming, Hann) attenuate high
    frequencies at the cost of some spatial resolution:

    - **Ramp**: Best resolution, most noise
    - **Shepp-Logan**: Good balance
    - **Cosine**: Smoother
    - **Hamming/Hann**: Smoothest, least noise

    **Angular Sampling:**

    For good reconstruction quality, angles should span at least :math:`[0, \\pi)`.
    The number of angles should be comparable to the number of detector bins
    (Nyquist criterion suggests ``num_angles >= pi/2 * num_bins``).

    **Relation to Radon Transform:**

    For sufficient angular sampling and noise-free data:

    .. math::

        \\text{inverse\\_radon\\_transform}(\\text{radon\\_transform}(f)) \\approx f

    See Also
    --------
    radon_transform : Forward Radon transform.
    """
    if filter_type not in _FILTER_TYPE_MAP:
        raise ValueError(
            f"filter_type must be one of {list(_FILTER_TYPE_MAP.keys())}, "
            f"got '{filter_type}'"
        )

    filter_int = _FILTER_TYPE_MAP[filter_type]
    output_size_int = output_size if output_size is not None else -1

    return torch.ops.torchscience.inverse_radon_transform(
        input,
        angles,
        circle,
        output_size_int,
        filter_int,
    )
