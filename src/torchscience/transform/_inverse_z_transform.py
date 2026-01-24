"""Inverse Z-transform implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_z_transform"]


def inverse_z_transform(
    input: Tensor,
    n: Tensor,
    z: Tensor,
    *,
    dim: int = -1,
) -> Tensor:
    r"""Compute the inverse Z-transform.

    The inverse Z-transform recovers the discrete-time sequence :math:`x[n]`
    from its Z-transform :math:`X(z)`:

    .. math::

        x[n] = \frac{1}{2\pi j} \oint_C X(z) z^{n-1} dz

    This implementation uses numerical evaluation on a contour (typically
    the unit circle) combined with the inverse DFT relationship.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`X(z)` sampled at points ``z``.
    n : Tensor
        Sample indices where to evaluate the inverse transform.
        Should be integers (will be cast to float for computation).
    z : Tensor
        Complex values where input is sampled.
        For best results, sample uniformly on a circle in the complex plane.
    dim : int, optional
        Dimension along which to transform. Default: -1.

    Returns
    -------
    Tensor
        Inverse Z-transform :math:`x[n]` evaluated at the given sample indices.
        The dimension ``dim`` is replaced by the shape of ``n``.

    Examples
    --------
    Round-trip with Z-transform:

    >>> import torch
    >>> import torchscience.transform as T
    >>> # Original sequence
    >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    >>> N = len(x)
    >>> # Sample on unit circle (uniformly spaced)
    >>> k = torch.arange(N, dtype=torch.float64)
    >>> z = torch.exp(2j * torch.pi * k / N)
    >>> # Forward transform
    >>> X = T.z_transform(x, z)
    >>> # Inverse transform
    >>> n = torch.arange(N, dtype=torch.float64)
    >>> x_reconstructed = T.inverse_z_transform(X, n, z)

    Notes
    -----
    When the Z-transform is sampled uniformly on the unit circle, the
    inverse Z-transform reduces to the inverse DFT (up to scaling).

    For general contours, this implementation uses numerical integration.
    The accuracy depends on the sampling density and the choice of contour.

    See Also
    --------
    z_transform : Forward Z-transform.
    inverse_fourier_transform : Inverse continuous Fourier transform.
    """
    return torch.ops.torchscience.inverse_z_transform(input, n, z, dim)
