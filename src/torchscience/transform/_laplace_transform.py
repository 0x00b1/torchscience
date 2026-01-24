"""Numerical Laplace transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Integration method mapping
_INTEGRATION_METHOD_MAP = {
    "trapezoidal": 0,
    "simpson": 1,
    "gauss_legendre": 2,
}


def laplace_transform(
    input: Tensor,
    s: Tensor,
    t: Tensor,
    *,
    dim: int = -1,
    integration_method: Literal[
        "trapezoidal", "simpson", "gauss_legendre"
    ] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical Laplace transform.

    The Laplace transform of a function :math:`f(t)` is defined as:

    .. math::
        \mathcal{L}\{f(t)\}(s) = F(s) = \int_0^\infty f(t) e^{-st} dt

    This function computes a numerical approximation using quadrature methods.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`f(t)` sampled at time points ``t``.
    s : Tensor
        Complex frequency values where to evaluate the transform.
        Can be real (for real Laplace transform) or complex.
    t : Tensor
        Time points where ``input`` is sampled. Must be 1-D, non-negative,
        and sorted in ascending order. The first element should typically be 0.
    dim : int, optional
        Dimension of ``input`` along which the function is sampled.
        Default: ``-1`` (last dimension).
    integration_method : str, optional
        Numerical integration method:

        - ``'trapezoidal'``: Trapezoidal rule (default). Works for non-uniform spacing.
        - ``'simpson'``: Simpson's rule. More accurate for smooth functions with uniform spacing.
        - ``'gauss_legendre'``: Gauss-Legendre quadrature (falls back to trapezoidal).

        Default: ``'trapezoidal'``.

    Returns
    -------
    Tensor
        The Laplace transform :math:`F(s)` evaluated at the given ``s`` values.
        The shape is ``input.shape`` with dimension ``dim`` replaced by ``s.shape``.

    Examples
    --------
    Laplace transform of exponential decay :math:`f(t) = e^{-at}`:

    >>> t = torch.linspace(0, 10, 1000, dtype=torch.float64)
    >>> a = 2.0
    >>> f = torch.exp(-a * t)
    >>> s = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    >>> F = laplace_transform(f, s, t)
    >>> # Analytical: F(s) = 1 / (s + a)
    >>> analytical = 1.0 / (s + a)
    >>> torch.allclose(F, analytical, rtol=1e-2)
    True

    Transform of a step function :math:`f(t) = 1`:

    >>> f = torch.ones(1000, dtype=torch.float64)
    >>> F = laplace_transform(f, s, t)
    >>> # Analytical: F(s) = 1/s
    >>> analytical = 1.0 / s
    >>> torch.allclose(F, analytical, rtol=1e-2)
    True

    Batched transform:

    >>> # Multiple functions
    >>> f_batch = torch.randn(5, 1000, dtype=torch.float64)
    >>> F_batch = laplace_transform(f_batch, s, t, dim=-1)
    >>> F_batch.shape
    torch.Size([5, 3])

    Notes
    -----
    **Accuracy:**

    The accuracy depends on:

    1. Sampling density in ``t`` (more points = higher accuracy)
    2. Integration method (Simpson's is more accurate for smooth functions)
    3. Truncation of the integral at ``t[-1]`` instead of infinity

    For best results:

    - Use a large enough final time so that :math:`f(t) e^{-\Re(s) t}` is negligible
    - Use enough sample points for smooth representation
    - Choose ``s`` with positive real part for convergence

    **Relation to other transforms:**

    - When :math:`s = i\omega` (purely imaginary), this becomes the Fourier transform
    - The bilateral (two-sided) Laplace transform integrates from :math:`-\infty` to :math:`+\infty`

    See Also
    --------
    inverse_laplace_transform : Inverse Laplace transform.
    two_sided_laplace_transform : Bilateral Laplace transform.
    scipy.integrate.quad : General numerical integration.
    """
    if integration_method not in _INTEGRATION_METHOD_MAP:
        raise ValueError(
            f"integration_method must be 'trapezoidal', 'simpson', or 'gauss_legendre', "
            f"got {integration_method}"
        )

    method_int = _INTEGRATION_METHOD_MAP[integration_method]

    return torch.ops.torchscience.laplace_transform(
        input,
        s,
        t,
        dim,
        method_int,
    )
