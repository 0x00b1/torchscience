import torch
from torch import Tensor


def beta(a: Tensor, b: Tensor) -> Tensor:
    r"""
    Beta function.

    Computes the beta function B(a, b) evaluated at each element.

    Mathematical Definition
    -----------------------
    The beta function is defined as:

    .. math::

       B(a, b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)}

    Equivalently, it can be expressed as the integral:

    .. math::

       B(a, b) = \int_0^1 t^{a-1} (1-t)^{b-1} \, dt

    for Re(a) > 0 and Re(b) > 0.

    The function satisfies the symmetry relation:

    .. math::

       B(a, b) = B(b, a)

    Domain
    ------
    - a, b: positive real numbers or complex numbers with positive real part
    - Poles occur when a or b is a non-positive integer (returns inf)

    Special Values
    --------------
    - B(1, 1) = 1
    - B(1, n) = 1/n for positive integer n
    - B(0.5, 0.5) = pi
    - B(a, b) = B(b, a) (symmetry)
    - B(a, 1) = 1/a
    - B(1, b) = 1/b

    Algorithm
    ---------
    Computed as exp(log_beta(a, b)) where log_beta uses the log-gamma function
    for numerical stability:

    .. math::

       \log B(a, b) = \log \Gamma(a) + \log \Gamma(b) - \log \Gamma(a + b)

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    The beta function appears in many mathematical and statistical contexts:
    - Normalization constant for the beta distribution
    - Binomial coefficients: C(n, k) = 1 / ((n+1) * B(n-k+1, k+1))
    - Relationship to factorials: B(m, n) = (m-1)!(n-1)! / (m+n-1)!
    - Probability theory and Bayesian inference
    - String theory and physics

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are computed using the digamma function:

    .. math::

       \frac{\partial B}{\partial a} &= B(a, b) \left[ \psi(a) - \psi(a+b) \right] \\
       \frac{\partial B}{\partial b} &= B(a, b) \left[ \psi(b) - \psi(a+b) \right]

    where :math:`\psi(x) = \frac{d}{dx} \ln \Gamma(x)` is the digamma function.

    Second-order derivatives (gradgradcheck) are also supported, using the
    trigamma function:

    .. math::

       \frac{\partial^2 B}{\partial a^2} &= B(a, b) \left[ (\psi(a) - \psi(a+b))^2 + \psi'(a) - \psi'(a+b) \right] \\
       \frac{\partial^2 B}{\partial b^2} &= B(a, b) \left[ (\psi(b) - \psi(a+b))^2 + \psi'(b) - \psi'(a+b) \right] \\
       \frac{\partial^2 B}{\partial a \partial b} &= B(a, b) \left[ (\psi(a) - \psi(a+b))(\psi(b) - \psi(a+b)) - \psi'(a+b) \right]

    where :math:`\psi'(x)` is the trigamma function.

    Parameters
    ----------
    a : Tensor
        First parameter. Must be positive (a > 0 for real, Re(a) > 0 for complex).
        Broadcasting with b is supported.
    b : Tensor
        Second parameter. Must be positive (b > 0 for real, Re(b) > 0 for complex).
        Broadcasting with a is supported.

    Returns
    -------
    Tensor
        The beta function B(a, b) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0, 2.0, 3.0])
    >>> b = torch.tensor([1.0, 2.0, 3.0])
    >>> beta(a, b)
    tensor([1.0000, 0.1667, 0.0333])

    B(0.5, 0.5) = pi:

    >>> a = torch.tensor([0.5])
    >>> b = torch.tensor([0.5])
    >>> beta(a, b)
    tensor([3.1416])

    Symmetry B(a, b) = B(b, a):

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([5.0])
    >>> beta(a, b)
    tensor([0.0333])
    >>> beta(b, a)
    tensor([0.0333])

    Complex input:

    >>> a = torch.tensor([1.0 + 1.0j])
    >>> b = torch.tensor([2.0 + 0.5j])
    >>> beta(a, b)
    tensor([0.1520-0.1854j])

    Autograd:

    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> b = torch.tensor([3.0])
    >>> y = beta(a, b)
    >>> y.backward()
    >>> a.grad
    tensor([-0.0528])

    .. warning:: Overflow for small arguments

       The beta function can overflow for small positive arguments since
       Gamma(a) and Gamma(b) can be very large:

       - B(0.01, 0.01) involves Gamma(0.01) ~ 99.4
       - For very small a or b, use log-beta instead: torch.special.betaln

    .. warning:: Poles at non-positive integers

       The function returns inf when a or b is a non-positive integer
       (0, -1, -2, ...) since these are poles of the gamma function.

    Notes
    -----
    - For numerical stability with small arguments, consider using
      ``torch.special.betaln`` (log-beta) and exponentiating only when needed.
    - The implementation uses the identity B(a,b) = exp(log_beta(a,b)) where
      log_beta is computed via log-gamma functions.

    See Also
    --------
    torch.special.betaln : Natural logarithm of the beta function (preferred for small arguments)
    gamma : Gamma function
    incomplete_beta : Regularized incomplete beta function
    """
    return torch.ops.torchscience.beta(a, b)
