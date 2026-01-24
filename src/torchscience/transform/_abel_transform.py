"""Abel transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["abel_transform"]

# Integration method mapping
_INTEGRATION_METHOD_MAP = {
    "trapezoidal": 0,
    "simpson": 1,
}


def abel_transform(
    input: Tensor,
    y: Tensor,
    r: Tensor,
    *,
    dim: int = -1,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical Abel transform.

    The Abel transform of a radially symmetric function :math:`f(r)` is:

    .. math::

        F(y) = 2 \int_y^\infty \frac{f(r) \, r}{\sqrt{r^2 - y^2}} \, dr

    This integral represents the line-of-sight integration through a
    cylindrically symmetric object, commonly used in:

    - Plasma diagnostics
    - Flame spectroscopy
    - CT reconstruction of radially symmetric objects

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`f(r)` sampled at radial points ``r``.
    y : Tensor
        Impact parameter values where to evaluate the transform.
        Should satisfy :math:`0 \leq y \leq \max(r)`.
    r : Tensor
        Radial points where input is sampled.
        Must be positive and sorted in ascending order.
    dim : int, optional
        Dimension along which to integrate. Default: -1.
    integration_method : {"trapezoidal", "simpson"}, optional
        Numerical integration method. Default: "trapezoidal".

    Returns
    -------
    Tensor
        Abel transform :math:`F(y)` evaluated at the given ``y`` values.
        The dimension ``dim`` is replaced by the shape of ``y``.

    Examples
    --------
    Abel transform of a Gaussian:

    >>> import torch
    >>> import torchscience.transform as T
    >>> r = torch.linspace(0.01, 10, 1000, dtype=torch.float64)
    >>> # Gaussian: f(r) = exp(-r^2)
    >>> f = torch.exp(-r**2)
    >>> y = torch.linspace(0, 5, 100, dtype=torch.float64)
    >>> F = T.abel_transform(f, y, r)
    >>> # For Gaussian: F(y) = sqrt(pi) * exp(-y^2)

    Notes
    -----
    The Abel transform arises from integrating a radially symmetric 3D
    function along parallel lines (line-of-sight integration).

    The transform has a singularity at :math:`r = y`, which requires
    careful numerical handling. This implementation uses regularization
    to handle the singularity.

    See Also
    --------
    inverse_abel_transform : Inverse Abel transform.
    radon_transform : General tomographic projection.
    """
    if integration_method not in _INTEGRATION_METHOD_MAP:
        raise ValueError(
            f"integration_method must be 'trapezoidal' or 'simpson', "
            f"got {integration_method}"
        )

    method_int = _INTEGRATION_METHOD_MAP[integration_method]

    return torch.ops.torchscience.abel_transform(input, y, r, dim, method_int)
