import torch
from torch import Tensor


def chebyshev_polynomial_u(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Chebyshev polynomial of the second kind.

    Computes the Chebyshev polynomial of the second kind U_n(x).

    Mathematical Definition
    -----------------------
    For integer n >= 0, the Chebyshev polynomial of the second kind is defined by:

    .. math::

       U_n(x) = \frac{\sin((n+1) \arccos(x))}{\sin(\arccos(x))}
              = \frac{\sin((n+1) \theta)}{\sin(\theta)}

    where :math:`\theta = \arccos(x)`.

    The recurrence relation is:

    .. math::

       U_0(x) &= 1 \\
       U_1(x) &= 2x \\
       U_n(x) &= 2x \, U_{n-1}(x) - U_{n-2}(x)

    For |x| > 1, the hyperbolic continuation is used:

    .. math::

       U_n(x) = \frac{\sinh((n+1) \eta)}{\sinh(\eta)}

    where :math:`\eta = \mathrm{arccosh}(|x|)` and appropriate sign adjustments
    are made for x < -1.

    Special Values
    --------------
    - U_0(x) = 1 for all x
    - U_1(x) = 2x for all x
    - U_n(1) = n + 1 for all integer n >= 0
    - U_n(-1) = (-1)^n * (n + 1) for all integer n >= 0
    - U_n(0) = cos(n * pi / 2) for integer n (i.e., 1, 0, -1, 0, 1, ...)

    Domain
    ------
    - n: non-negative real value (integral values use efficient recurrence)
    - x: any real value
    - For |x| <= 1: uses trigonometric formula
    - For |x| > 1: uses hyperbolic continuation

    Relationship to Chebyshev T
    ---------------------------
    The Chebyshev polynomials of the first and second kind are related by:

    .. math::

       \frac{d}{dx} T_n(x) = n \, U_{n-1}(x)

    and

    .. math::

       T_n(x) = U_n(x) - x \, U_{n-1}(x)

    Applications
    ------------
    Chebyshev polynomials of the second kind appear in:
    - Spectral methods for differential equations
    - Polynomial interpolation and approximation
    - Numerical integration (Gauss-Chebyshev quadrature of the second kind)
    - Filter design

    Autograd Support
    ----------------
    - Gradients for x are computed when x.requires_grad is True.
    - Gradients for n are always zero (n is treated as discrete).
    - Second-order derivatives are supported for most domains.

    Backward formula:

    .. math::

       \frac{\partial U_n(x)}{\partial x} = \frac{(n+1) T_{n+1}(x) - x U_n(x)}{x^2 - 1}

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Should be non-negative.
        When integral, uses efficient polynomial recurrence.
    x : Tensor
        Input tensor. Can be any real value.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Chebyshev polynomial U_n(x) evaluated at the input values.

    Examples
    --------
    Integer degree with real input:

    >>> n = torch.tensor([0, 1, 2, 3])
    >>> x = torch.tensor([0.5])
    >>> chebyshev_polynomial_u(n, x)
    tensor([ 1.0000,  1.0000,  0.0000, -1.0000])

    At x = 1:

    >>> n = torch.tensor([0, 1, 2, 3, 4])
    >>> x = torch.tensor([1.0])
    >>> chebyshev_polynomial_u(n, x)  # Returns n + 1
    tensor([1., 2., 3., 4., 5.])

    Polynomial values:

    >>> n = torch.tensor([2.0])
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> chebyshev_polynomial_u(n, x)  # U_2(x) = 4x^2 - 1
    tensor([-1.,  0.,  3.])

    With gradients:

    >>> n = torch.tensor([2.0])
    >>> x = torch.tensor([0.5], requires_grad=True)
    >>> y = chebyshev_polynomial_u(n, x)
    >>> y.backward()
    >>> x.grad
    tensor([4.])

    .. warning:: Gradient singularity

       The gradient has singularities at x = +/- 1. Gradients at these points
       may be inf or NaN.

    See Also
    --------
    chebyshev_polynomial_t : Chebyshev polynomial of the first kind
    """
    return torch.ops.torchscience.chebyshev_polynomial_u(x, n)
