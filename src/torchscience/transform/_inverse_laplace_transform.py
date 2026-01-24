"""Inverse Laplace transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_laplace_transform"]

# Integration method mapping
_INTEGRATION_METHOD_MAP = {
    "trapezoidal": 0,
    "simpson": 1,
}


def inverse_laplace_transform(
    input: Tensor,
    t: Tensor,
    s: Tensor,
    *,
    dim: int = -1,
    sigma: float | None = None,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical inverse Laplace transform.

    The inverse Laplace transform is defined as:

    .. math::
        \mathcal{L}^{-1}\{F(s)\}(t) = f(t) = \frac{1}{2\pi i}
        \int_{\sigma - i\infty}^{\sigma + i\infty} F(s) e^{st} ds

    This is computed using the Bromwich integral, which can be rewritten as:

    .. math::
        f(t) = \frac{e^{\sigma t}}{2\pi}
        \int_{-\infty}^{\infty} F(\sigma + i\omega) e^{i\omega t} d\omega

    This function computes a numerical approximation using quadrature.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`F(s)` sampled at frequency points ``s``.
    t : Tensor
        Time points where to evaluate the inverse transform.
    s : Tensor
        Complex frequency values :math:`s = \sigma + i\omega` where ``input``
        is sampled. Should be along a vertical line in the complex plane
        (constant real part).
    dim : int, optional
        Dimension of ``input`` along which the function is sampled.
        Default: ``-1`` (last dimension).
    sigma : float, optional
        Real part of the Bromwich contour. If None, uses the real part of
        the first element of ``s``. Default: ``None``.
    integration_method : str, optional
        Numerical integration method:

        - ``'trapezoidal'``: Trapezoidal rule (default).
        - ``'simpson'``: Simpson's rule.

        Default: ``'trapezoidal'``.

    Returns
    -------
    Tensor
        The inverse Laplace transform :math:`f(t)` evaluated at the given ``t`` values.
        The shape is ``input.shape`` with dimension ``dim`` replaced by ``t.shape``.

    Examples
    --------
    Inverse Laplace transform of :math:`F(s) = 1/(s+a)`:

    >>> import torch
    >>> import torchscience.transform as T
    >>> a = 2.0
    >>> omega = torch.linspace(-50, 50, 1001, dtype=torch.float64)
    >>> sigma = 3.0  # Must be > a for convergence
    >>> s = sigma + 1j * omega
    >>> F = 1.0 / (s + a)  # Laplace transform of exp(-a*t)
    >>> t = torch.linspace(0, 2, 100, dtype=torch.float64)
    >>> f = T.inverse_laplace_transform(F, t, s, sigma=sigma)
    >>> # Should approximate exp(-a*t)

    Notes
    -----
    **Accuracy:**

    The accuracy depends on:

    1. The range of ``omega`` (imaginary part of ``s``) - should cover
       the significant frequency content
    2. The sampling density in ``s``
    3. The choice of ``sigma`` - should be to the right of all singularities

    **Choosing sigma:**

    The contour must lie to the right of all singularities of ``F(s)`` in
    the complex plane. If ``F(s)`` has poles at :math:`s_1, s_2, \ldots`,
    choose :math:`\sigma > \max_i \text{Re}(s_i)`.

    See Also
    --------
    laplace_transform : Forward Laplace transform.
    """
    if integration_method not in _INTEGRATION_METHOD_MAP:
        raise ValueError(
            f"integration_method must be 'trapezoidal' or 'simpson', "
            f"got {integration_method}"
        )

    method_int = _INTEGRATION_METHOD_MAP[integration_method]

    # Determine sigma from s if not provided
    if sigma is None:
        sigma = s.flatten()[0].real.item()

    return torch.ops.torchscience.inverse_laplace_transform(
        input,
        t,
        s,
        dim,
        sigma,
        method_int,
    )
