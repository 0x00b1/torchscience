import torch
from torch import Tensor


def log_beta(a: Tensor, b: Tensor) -> Tensor:
    r"""
    Natural logarithm of the beta function.

    Computes the natural logarithm of the beta function evaluated at each
    element. This is numerically more stable than computing log(beta(a, b))
    directly, especially for small or large arguments where the beta function
    may overflow or underflow.

    Mathematical Definition
    -----------------------
    The log-beta function is defined as:

    .. math::

       \log B(a, b) = \log \Gamma(a) + \log \Gamma(b) - \log \Gamma(a + b)

    This is related to the beta function by:

    .. math::

       \log B(a, b) = \log \left( \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)} \right)

    The function satisfies the symmetry relation:

    .. math::

       \log B(a, b) = \log B(b, a)

    Domain
    ------
    - a, b: positive real numbers or complex numbers with positive real part
    - Returns -inf when a or b is a non-positive integer (poles of gamma)

    Special Values
    --------------
    - log B(1, 1) = 0 (since B(1,1) = 1)
    - log B(1, n) = -log(n) for positive integer n
    - log B(0.5, 0.5) = log(pi)
    - log B(a, b) = log B(b, a) (symmetry)
    - log B(a, 1) = -log(a)
    - log B(1, b) = -log(b)

    Algorithm
    ---------
    Computed as log_gamma(a) + log_gamma(b) - log_gamma(a + b) using the
    log-gamma function for numerical stability.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    The log-beta function is useful in:
    - Log-likelihood computations for beta and Dirichlet distributions
    - Numerical computations where beta function values would overflow/underflow
    - Log-space probability calculations in Bayesian inference
    - Binomial coefficients: log C(n, k) = -log(n+1) - log B(n-k+1, k+1)

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

       \frac{\partial \log B}{\partial a} &= \psi(a) - \psi(a+b) \\
       \frac{\partial \log B}{\partial b} &= \psi(b) - \psi(a+b)

    where :math:`\psi(x) = \frac{d}{dx} \ln \Gamma(x)` is the digamma function.

    Second-order derivatives are also supported, using the trigamma function:

    .. math::

       \frac{\partial^2 \log B}{\partial a^2} &= \psi'(a) - \psi'(a+b) \\
       \frac{\partial^2 \log B}{\partial b^2} &= \psi'(b) - \psi'(a+b) \\
       \frac{\partial^2 \log B}{\partial a \partial b} &= -\psi'(a+b)

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
        The natural logarithm of the beta function, log B(a, b).
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0, 2.0, 3.0])
    >>> b = torch.tensor([1.0, 2.0, 3.0])
    >>> log_beta(a, b)
    tensor([ 0.0000, -1.7918, -3.4012])

    Comparison with log(B(a,b)):

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> log_beta(a, b)
    tensor([-3.4012])
    >>> torch.log(torch.tensor([1/12]))  # B(2,3) = 1/12
    tensor([-2.4849])

    log B(0.5, 0.5) = log(pi):

    >>> a = torch.tensor([0.5])
    >>> b = torch.tensor([0.5])
    >>> log_beta(a, b)
    tensor([1.1447])
    >>> torch.log(torch.tensor([3.14159]))
    tensor([1.1447])

    Symmetry log B(a, b) = log B(b, a):

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([5.0])
    >>> log_beta(a, b)
    tensor([-4.7875])
    >>> log_beta(b, a)
    tensor([-4.7875])

    log B(1, n) = -log(n):

    >>> b = torch.tensor([2.0, 3.0, 4.0, 5.0])
    >>> log_beta(torch.ones_like(b), b)
    tensor([-0.6931, -1.0986, -1.3863, -1.6094])
    >>> -torch.log(b)
    tensor([-0.6931, -1.0986, -1.3863, -1.6094])

    Complex input:

    >>> a = torch.tensor([1.0 + 1.0j])
    >>> b = torch.tensor([2.0 + 0.5j])
    >>> log_beta(a, b)
    tensor([-1.6541-0.8845j])

    Autograd:

    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> b = torch.tensor([3.0])
    >>> y = log_beta(a, b)
    >>> y.backward()
    >>> a.grad  # psi(2) - psi(5) = 0.4228 - 1.5061 = -1.0833
    tensor([-1.0833])

    Notes
    -----
    - This function is preferred over log(beta(a,b)) for numerical stability,
      especially when a or b are small (< 1) or large (> 100).
    - For very small a or b, the beta function can overflow, but log_beta
      remains well-defined and finite.
    - The relationship beta(a,b) = exp(log_beta(a,b)) holds mathematically,
      but computing beta directly may overflow.

    See Also
    --------
    beta : Beta function B(a, b)
    gamma : Gamma function
    torch.special.gammaln : Log-gamma function (PyTorch built-in)
    """
    return torch.ops.torchscience.log_beta(a, b)
