"""Inverse Abel transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_abel_transform"]

# Integration method mapping
_INTEGRATION_METHOD_MAP = {
    "trapezoidal": 0,
    "simpson": 1,
}


def inverse_abel_transform(
    input: Tensor,
    r: Tensor,
    y: Tensor,
    *,
    dim: int = -1,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical inverse Abel transform.

    The inverse Abel transform recovers the radial function :math:`f(r)`
    from its Abel transform :math:`F(y)`:

    .. math::

        f(r) = -\frac{1}{\pi} \int_r^\infty
        \frac{dF/dy}{\sqrt{y^2 - r^2}} \, dy

    This is used to reconstruct radially symmetric distributions from
    line-of-sight measurements.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`F(y)` sampled at impact parameter points ``y``.
    r : Tensor
        Radial points where to evaluate the inverse transform.
    y : Tensor
        Impact parameter points where input is sampled.
        Must be non-negative and sorted in ascending order.
    dim : int, optional
        Dimension along which to integrate. Default: -1.
    integration_method : {"trapezoidal", "simpson"}, optional
        Numerical integration method. Default: "trapezoidal".

    Returns
    -------
    Tensor
        Inverse Abel transform :math:`f(r)` evaluated at the given ``r`` values.
        The dimension ``dim`` is replaced by the shape of ``r``.

    Examples
    --------
    Round-trip with Abel transform:

    >>> import torch
    >>> import torchscience.transform as T
    >>> r = torch.linspace(0.01, 10, 500, dtype=torch.float64)
    >>> f = torch.exp(-r**2)  # Gaussian
    >>> y = torch.linspace(0.01, 10, 500, dtype=torch.float64)
    >>> F = T.abel_transform(f, y, r)
    >>> f_reconstructed = T.inverse_abel_transform(F, r, y)

    Notes
    -----
    The inverse Abel transform involves computing the derivative of the
    input, which can amplify noise. For noisy data, consider smoothing
    before applying the inverse transform.

    This implementation uses numerical differentiation followed by
    integration, with regularization to handle the singularity at
    :math:`y = r`.

    See Also
    --------
    abel_transform : Forward Abel transform.
    inverse_radon_transform : General tomographic reconstruction.
    """
    if integration_method not in _INTEGRATION_METHOD_MAP:
        raise ValueError(
            f"integration_method must be 'trapezoidal' or 'simpson', "
            f"got {integration_method}"
        )

    method_int = _INTEGRATION_METHOD_MAP[integration_method]

    return torch.ops.torchscience.inverse_abel_transform(
        input, r, y, dim, method_int
    )
