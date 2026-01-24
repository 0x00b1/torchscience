"""Z-transform implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["z_transform"]


def z_transform(
    input: Tensor,
    z: Tensor,
    *,
    dim: int = -1,
) -> Tensor:
    r"""Compute the Z-transform.

    The Z-transform of a discrete-time sequence :math:`x[n]` is:

    .. math::

        X(z) = \sum_{n=0}^{N-1} x[n] z^{-n}

    This is the discrete-time analog of the Laplace transform, commonly
    used in digital signal processing and control systems.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`x[n]` representing a discrete-time sequence.
    z : Tensor
        Complex values where to evaluate the transform.
        Can be complex128 or complex64.
    dim : int, optional
        Dimension along which to transform. Default: -1.

    Returns
    -------
    Tensor
        Z-transform :math:`X(z)` evaluated at the given ``z`` values.
        The dimension ``dim`` is replaced by the shape of ``z``.

    Examples
    --------
    Z-transform of a geometric sequence:

    >>> import torch
    >>> import torchscience.transform as T
    >>> # x[n] = a^n for n = 0, 1, ..., N-1
    >>> a = 0.9
    >>> N = 100
    >>> n = torch.arange(N, dtype=torch.float64)
    >>> x = a ** n
    >>> # Evaluate on the unit circle (DTFT)
    >>> omega = torch.linspace(0, 2 * torch.pi, 256, dtype=torch.float64)
    >>> z = torch.exp(1j * omega)
    >>> X = T.z_transform(x, z)
    >>> # Analytical: X(z) = z / (z - a) for |z| > |a|

    Frequency response on the unit circle:

    >>> # For a causal FIR filter h[n]
    >>> h = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    >>> omega = torch.linspace(0, torch.pi, 100, dtype=torch.float64)
    >>> z_unit = torch.exp(1j * omega)
    >>> H = T.z_transform(h, z_unit)
    >>> magnitude_response = torch.abs(H)

    Notes
    -----
    When evaluated on the unit circle :math:`z = e^{j\omega}`, the Z-transform
    becomes the Discrete-Time Fourier Transform (DTFT).

    The region of convergence (ROC) depends on the sequence:
    - Right-sided sequences: ROC is outside a circle |z| > R
    - Left-sided sequences: ROC is inside a circle |z| < R
    - Two-sided sequences: ROC is an annulus R1 < |z| < R2

    See Also
    --------
    inverse_z_transform : Inverse Z-transform.
    fourier_transform : Continuous-time Fourier transform.
    """
    return torch.ops.torchscience.z_transform(input, z, dim)
