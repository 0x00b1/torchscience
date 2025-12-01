import torch
from torch import Tensor


def incomplete_beta(z: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""
    Regularized incomplete beta function.

    Computes the regularized incomplete beta function I_z(a, b).

    Mathematical Definition
    -----------------------
    The regularized incomplete beta function is defined as:

    .. math::

       I_z(a, b) = \frac{B_z(a, b)}{B(a, b)}

    where:

    .. math::

       B_z(a, b) = \int_0^z t^{a-1} (1-t)^{b-1} \, dt

    is the incomplete beta function, and:

    .. math::

       B(a, b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}

    is the beta function.

    The function satisfies the symmetry relation:

    .. math::

       I_z(a, b) + I_{1-z}(b, a) = 1

    Domain
    ------
    - z: Real z in [0, 1], or complex z (extended domain via analytic continuation)
    - a: must be positive (a > 0, or Re(a) > 0 for complex)
    - b: must be positive (b > 0, or Re(b) > 0 for complex)

    For real z outside [0, 1] or non-positive a, b, the result is NaN.

    For complex z with ``|z| >= 1``, uses analytic continuation via the
    hypergeometric 2F1 relation:

    .. math::

       I_z(a, b) = \frac{z^a}{a \, B(a,b)} \, {}_2F_1(a, 1-b; a+1; z)

    Two regions are handled:

    - Region B: ``|z| >= 1``, ``|1-z| < 1`` -- uses symmetry I_z(a,b) = 1 - I_{1-z}(b,a)
    - Region C: ``|z| > 1``, ``|1-z| >= 1`` -- uses hypergeometric linear transformation

    Algorithm
    ---------
    Uses a continued fraction expansion for numerical evaluation:
    - When z < (a + 1) / (a + b + 2), uses direct continued fraction
    - Otherwise, uses symmetry: I_z(a, b) = 1 - I_{1-z}(b, a)

    The continued fraction is evaluated using a modified Lentz's algorithm
    for numerical stability.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Special Values
    --------------
    - I_0(a, b) = 0
    - I_1(a, b) = 1
    - I_z(1, 1) = z (uniform distribution CDF)
    - I_z(1, b) = 1 - (1-z)^b
    - I_z(a, 1) = z^a

    Applications
    ------------
    The regularized incomplete beta function is widely used in statistics:
    - Cumulative distribution function (CDF) of the beta distribution
    - CDF of the binomial distribution
    - CDF of the F-distribution
    - CDF of Student's t-distribution
    - Confidence intervals for binomial proportions

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Complex dtypes (complex64, complex128) are supported for all z via
      analytic continuation

    Autograd Support
    ----------------
    All first-order derivatives are computed analytically:

    .. math::

       \frac{\partial I}{\partial z} &= \frac{z^{a-1} (1-z)^{b-1}}{B(a, b)} \\
       \frac{\partial I}{\partial a} &= \frac{J_a}{B(a,b)} - I_z(a,b) [\psi(a) - \psi(a+b)] \\
       \frac{\partial I}{\partial b} &= \frac{J_b}{B(a,b)} - I_z(a,b) [\psi(b) - \psi(a+b)]

    where :math:`\psi` is the digamma function and :math:`J_a, J_b` are log-weighted
    integrals computed using 32-point Gauss-Legendre quadrature.

    All second-order derivatives are also fully analytical using:

    - Trigamma functions :math:`\psi'(x)` for parameter second derivatives
    - Doubly log-weighted integrals :math:`K_{aa}, K_{ab}, K_{bb}` via quadrature

    The analytical formulas for parameter second derivatives are:

    .. math::

       \frac{\partial^2 I}{\partial a^2} &= \frac{K_{aa}}{B} - \frac{2 J_a}{B}[\psi(a)-\psi(a+b)] + I_z[\psi(a)-\psi(a+b)]^2 - I_z[\psi'(a) - \psi'(a+b)] \\
       \frac{\partial^2 I}{\partial b^2} &= \frac{K_{bb}}{B} - \frac{2 J_b}{B}[\psi(b)-\psi(a+b)] + I_z[\psi(b)-\psi(a+b)]^2 - I_z[\psi'(b) - \psi'(a+b)] \\
       \frac{\partial^2 I}{\partial a \partial b} &= \frac{K_{ab}}{B} - \frac{J_a}{B}[\psi(b)-\psi(a+b)] - \frac{J_b}{B}[\psi(a)-\psi(a+b)] + I_z[\psi(a)-\psi(a+b)][\psi(b)-\psi(a+b)] + I_z \psi'(a+b)

    For complex inputs, Wirtinger derivative conventions are used:

    - First backward: grad_x = grad_output * conj(df/dx)
    - Double backward: d(grad_x)/dx_bar = grad_output * conj(d2f/dx2)

    Parameters
    ----------
    z : Tensor
        Input tensor. For real dtypes, values should be in [0, 1].
        For complex dtypes, values can be anywhere in the complex plane
        (analytic continuation is used for ``|z| >= 1``).
        Broadcasting with a and b is supported.
    a : Tensor
        First shape parameter. Must be positive (a > 0 for real, Re(a) > 0 for complex).
        Broadcasting with z and b is supported.
    b : Tensor
        Second shape parameter. Must be positive (b > 0 for real, Re(b) > 0 for complex).
        Broadcasting with z and a is supported.

    Returns
    -------
    Tensor
        The regularized incomplete beta function I_z(a, b) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([5.0])
    >>> incomplete_beta(z, a, b)
    tensor([0.0000, 0.6328, 0.8906, 0.9844, 1.0000])

    Special case I_z(1, 1) = z (uniform distribution):

    >>> z = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([1.0])
    >>> incomplete_beta(z, a, b)
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    Beta distribution CDF:

    >>> z = torch.tensor([0.5])
    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([2.0])
    >>> incomplete_beta(z, a, b)  # Symmetric beta, CDF at 0.5 should be 0.5
    tensor([0.5000])

    Autograd:

    >>> z = torch.tensor([0.3], requires_grad=True)
    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> y = incomplete_beta(z, a, b)
    >>> y.backward()
    >>> z.grad  # Gradient w.r.t. z
    tensor([1.7640])

    .. warning:: Reduced precision near boundaries

       For z very close to 0 or 1, numerical precision may be reduced.

    .. warning:: Slow convergence for extreme parameters

       For very large or very small values of a and b, the continued
       fraction may converge slowly or lose precision.

    .. warning:: Gradient accuracy for extreme parameters

       For extreme parameter values (a or b near 0 or very large), the
       log-weighted integrals used for gradients may lose accuracy.

    .. warning:: Finite difference gradients in Region C

       For complex z with ``|z| > 1`` in Region C (where ``|1-z| >= 1``),
       gradients use finite differences and may have reduced accuracy
       compared to the analytical gradients available in other regions.

    .. warning:: Regularization for near-integer (a - b)

       When a - b is close to an integer, the hypergeometric continuation
       uses a regularization technique which may have slightly reduced accuracy.

    Notes
    -----
    - The parameter order (z, a, b) matches the mathematical notation I_z(a, b)
      with the integration limit first.
    - For gradients with respect to a and b, the implementation uses 32-point
      Gauss-Legendre quadrature to compute log-weighted integrals, providing
      analytical (not finite-difference) gradients.
    - The symmetry relation I_z(a, b) = 1 - I_{1-z}(b, a) is used both for
      numerical stability and to extend the domain to complex z.

    See Also
    --------
    hypergeometric_2_f_1 : Gauss hypergeometric function (used internally for analytic continuation)
    gamma : Gamma function (used in beta function computation)
    torch.special.betaln : Natural logarithm of the beta function
    """
    return torch.ops.torchscience.incomplete_beta(z, a, b)
