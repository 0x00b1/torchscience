"""Inverse two-sided (bilateral) Laplace transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_two_sided_laplace_transform"]

# Integration method mapping
_INTEGRATION_METHOD_MAP = {
    "trapezoidal": 0,
    "simpson": 1,
}


def inverse_two_sided_laplace_transform(
    input: Tensor,
    t: Tensor,
    s: Tensor,
    *,
    dim: int = -1,
    sigma: float | None = None,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical inverse two-sided (bilateral) Laplace transform.

    The inverse two-sided Laplace transform is defined as:

    .. math::
        \mathcal{B}^{-1}\{F(s)\}(t) = f(t) = \frac{1}{2\pi i}
        \int_{\sigma - i\infty}^{\sigma + i\infty} F(s) e^{st} ds

    This is the same contour integral as the inverse unilateral Laplace
    transform, but the result may be non-zero for :math:`t < 0`.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`F(s)` sampled at frequency points ``s``.
    t : Tensor
        Time points where to evaluate the inverse transform.
        Can include negative values.
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
        The inverse two-sided Laplace transform :math:`f(t)` evaluated at
        the given ``t`` values, including negative times.

    Examples
    --------
    Inverse two-sided Laplace transform of a Gaussian in s-domain:

    >>> import torch
    >>> import torchscience.transform as T
    >>> omega = torch.linspace(-30, 30, 601, dtype=torch.float64)
    >>> sigma = 0.0  # For two-sided, can be in the ROC strip
    >>> s = sigma + 1j * omega
    >>> # sqrt(pi) * exp(s^2/4) is the two-sided Laplace transform of exp(-t^2)
    >>> import math
    >>> F = math.sqrt(math.pi) * torch.exp(s**2 / 4)
    >>> t = torch.linspace(-3, 3, 100, dtype=torch.float64)
    >>> f = T.inverse_two_sided_laplace_transform(F, t, s, sigma=sigma)
    >>> # Should approximate exp(-t^2)

    Notes
    -----
    **Choosing sigma:**

    For the two-sided Laplace transform, the contour must lie within the
    region of convergence (ROC), which is typically a vertical strip in
    the complex plane.

    See Also
    --------
    two_sided_laplace_transform : Forward two-sided Laplace transform.
    inverse_laplace_transform : Inverse unilateral Laplace transform.
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

    return torch.ops.torchscience.inverse_two_sided_laplace_transform(
        input,
        t,
        s,
        dim,
        sigma,
        method_int,
    )
