"""Inverse Mellin transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_mellin_transform"]

# Integration method mapping
_INTEGRATION_METHOD_MAP = {
    "trapezoidal": 0,
    "simpson": 1,
}


def inverse_mellin_transform(
    input: Tensor,
    t: Tensor,
    s: Tensor,
    *,
    dim: int = -1,
    c: float | None = None,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical inverse Mellin transform.

    The inverse Mellin transform is defined as:

    .. math::
        \mathcal{M}^{-1}\{F(s)\}(t) = f(t) = \frac{1}{2\pi i}
        \int_{c - i\infty}^{c + i\infty} F(s) t^{-s} ds

    This is computed using a contour integral along the vertical line
    :math:`\text{Re}(s) = c`, which can be rewritten as:

    .. math::
        f(t) = \frac{1}{2\pi}
        \int_{-\infty}^{\infty} F(c + i\omega) t^{-c - i\omega} d\omega

    This function computes a numerical approximation using quadrature.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`F(s)` sampled at frequency points ``s``.
    t : Tensor
        Points where to evaluate the inverse transform. Must be positive.
    s : Tensor
        Complex frequency values :math:`s = c + i\omega` where ``input``
        is sampled. Should be along a vertical line in the complex plane
        (constant real part).
    dim : int, optional
        Dimension of ``input`` along which the function is sampled.
        Default: ``-1`` (last dimension).
    c : float, optional
        Real part of the integration contour. If None, uses the real part of
        the first element of ``s``. Default: ``None``.
    integration_method : str, optional
        Numerical integration method:

        - ``'trapezoidal'``: Trapezoidal rule (default).
        - ``'simpson'``: Simpson's rule.

        Default: ``'trapezoidal'``.

    Returns
    -------
    Tensor
        The inverse Mellin transform :math:`f(t)` evaluated at the given ``t`` values.
        The shape is ``input.shape`` with dimension ``dim`` replaced by ``t.shape``.

    Examples
    --------
    Inverse Mellin transform of :math:`\Gamma(s)`:

    >>> import torch
    >>> import torchscience.transform as T
    >>> omega = torch.linspace(-20, 20, 401, dtype=torch.float64)
    >>> c = 1.0  # Must be in the fundamental strip
    >>> s = c + 1j * omega
    >>> # Gamma(s) is the Mellin transform of exp(-t)
    >>> F = torch.special.gammaln(s).exp()  # Gamma(s)
    >>> t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
    >>> f = T.inverse_mellin_transform(F, t, s, c=c)
    >>> # Should approximate exp(-t)

    Notes
    -----
    **Choosing c:**

    The contour must lie within the fundamental strip where :math:`F(s)`
    is analytic. For the Gamma function, any :math:`c > 0` works.

    **Relation to Laplace transform:**

    The Mellin transform is related to the two-sided Laplace transform by
    the substitution :math:`t = e^{-x}`.

    See Also
    --------
    mellin_transform : Forward Mellin transform.
    """
    if integration_method not in _INTEGRATION_METHOD_MAP:
        raise ValueError(
            f"integration_method must be 'trapezoidal' or 'simpson', "
            f"got {integration_method}"
        )

    method_int = _INTEGRATION_METHOD_MAP[integration_method]

    # Determine c from s if not provided
    if c is None:
        c = s.flatten()[0].real.item()

    return torch.ops.torchscience.inverse_mellin_transform(
        input,
        t,
        s,
        dim,
        c,
        method_int,
    )
