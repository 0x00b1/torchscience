import torch
from torch import Tensor


def log_gamma(z: Tensor) -> Tensor:
    r"""
    Logarithm of the gamma function.

    Computes the natural logarithm of the absolute value of the gamma function
    evaluated at each element of the input tensor.

    Mathematical Definition
    -----------------------
    The log-gamma function is defined as:

    .. math::

       \ln \Gamma(z) = \ln |\Gamma(z)|

    For positive real z, this equals ln(Gamma(z)). For complex z, this is
    the principal branch of the logarithm of the gamma function.

    Special Values
    --------------
    - ln(Gamma(1)) = ln(1) = 0
    - ln(Gamma(2)) = ln(1) = 0
    - ln(Gamma(n)) = ln((n-1)!) for positive integers n
    - ln(Gamma(1/2)) = ln(sqrt(pi)) = 0.5 * ln(pi)

    Domain
    ------
    - z: any real or complex value except non-positive integers
    - Returns +inf at z = 0, -1, -2, -3, ... (poles of gamma)

    Algorithm
    ---------
    Uses the Lanczos approximation with g=7 and n=9 coefficients.
    For z < 0.5, the reflection formula is applied:
    ln(Gamma(z)) = ln(pi) - ln(sin(pi*z)) - ln(Gamma(1-z))

    Advantages over gamma
    ---------------------
    - Avoids overflow for large arguments (gamma(171) overflows in float64)
    - More numerically stable for many applications
    - Directly useful in log-likelihood computations

    Applications
    ------------
    The log-gamma function is essential in:
    - Maximum likelihood estimation (log-likelihoods of gamma, beta, Dirichlet)
    - Bayesian inference (log prior/posterior computations)
    - Combinatorics (log of binomial coefficients for large n)
    - Statistical mechanics (partition functions)

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the digamma function:

    .. math::

       \frac{d}{dz} \ln \Gamma(z) = \psi(z) = \text{digamma}(z)

    Second-order derivatives are also supported using the trigamma function.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The log-gamma function evaluated at each element of z.

    Examples
    --------
    Evaluate at positive integers (ln of factorials):

    >>> z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> log_gamma(z)
    tensor([0.0000, 0.0000, 0.6931, 1.7918, 3.1781])

    Verify ln(Gamma(n)) = ln((n-1)!) for n=5:

    >>> import math
    >>> log_gamma(torch.tensor([5.0]))
    tensor([3.1781])  # ln(4!) = ln(24) = 3.178

    Avoid overflow for large arguments:

    >>> log_gamma(torch.tensor([200.0]))
    tensor([857.9337])  # gamma(200) would overflow

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j])
    >>> log_gamma(z)
    tensor([-0.6509+0.3019j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = log_gamma(z)
    >>> y.backward()
    >>> z.grad  # digamma(2) = 1 - gamma ≈ 0.4228
    tensor([0.4228])

    See Also
    --------
    torchscience.special_functions.gamma : Gamma function
    torchscience.special_functions.digamma : Digamma function (derivative)
    torch.special.gammaln : PyTorch's log-gamma implementation
    """
    return torch.ops.torchscience.log_gamma(z)
