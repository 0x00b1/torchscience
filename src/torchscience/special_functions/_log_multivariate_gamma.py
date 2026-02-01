import torch
from torch import Tensor


def log_multivariate_gamma(a: Tensor, d: int) -> Tensor:
    r"""
    Logarithm of the multivariate gamma function.

    The multivariate gamma function is a generalization of the gamma function
    used in the density function of the Wishart and inverse Wishart distributions.
    This function computes its natural logarithm.

    Mathematical Definition
    -----------------------
    The multivariate gamma function of dimension d is defined as:

    .. math::

        \Gamma_d(a) = \pi^{d(d-1)/4} \prod_{j=1}^{d} \Gamma\left(a + \frac{1-j}{2}\right)

    Taking the logarithm:

    .. math::

        \ln \Gamma_d(a) = \frac{d(d-1)}{4} \ln \pi + \sum_{j=1}^{d} \ln \Gamma\left(a + \frac{1-j}{2}\right)

    Domain
    ------
    Valid for a > (d-1)/2

    For the function to be well-defined, each argument to the gamma function
    must be positive:
    - a + (1-j)/2 > 0 for all j from 1 to d
    - The most restrictive constraint is when j=d: a + (1-d)/2 > 0
    - This gives: a > (d-1)/2

    Special Cases
    -------------
    - d=1: equals log_gamma(a)
    - d=2: equals 0.25*log(pi) + log_gamma(a) + log_gamma(a - 0.5)
    - d=3: equals 0.75*log(pi) + log_gamma(a) + log_gamma(a - 0.5) + log_gamma(a - 1)

    Applications
    ------------
    The multivariate gamma function appears in:
    - The normalizing constant of the Wishart distribution
    - The normalizing constant of the inverse Wishart distribution
    - Multivariate statistical analysis
    - Random matrix theory

    Autograd Support
    ----------------
    Gradients are fully supported when a.requires_grad is True.
    The gradient with respect to a is:

    .. math::

        \frac{d}{da} \ln \Gamma_d(a) = \sum_{j=1}^{d} \psi\left(a + \frac{1-j}{2}\right)

    where psi is the digamma function.

    Second-order derivatives are also supported.

    Note: The dimension parameter d does not support gradients as it is
    a discrete parameter.

    Parameters
    ----------
    a : Tensor
        Input tensor. Must have a > (d-1)/2 for valid results.
        Can be any floating-point dtype.
    d : int
        Dimension parameter. Must be a positive integer (d >= 1).

    Returns
    -------
    Tensor
        The log multivariate gamma function evaluated at each element of a
        with dimension d. Same shape and dtype as input a.

    Examples
    --------
    Evaluate at a scalar value with d=1 (should equal log_gamma):

    >>> import torch
    >>> from torchscience.special_functions import log_multivariate_gamma, log_gamma
    >>> a = torch.tensor([3.0])
    >>> log_multivariate_gamma(a, 1)
    tensor([0.6931])  # equals log_gamma(3.0) = log(2!)
    >>> log_gamma(a)
    tensor([0.6931])

    Evaluate with d=2:

    >>> a = torch.tensor([2.0], dtype=torch.float64)
    >>> log_multivariate_gamma(a, 2)
    tensor([0.6858], dtype=torch.float64)

    Comparison with scipy:

    >>> # scipy.special.multigammaln(2.0, 2) returns 0.6858...
    >>> import scipy.special
    >>> scipy.special.multigammaln(2.0, 2)
    0.6858...

    Autograd:

    >>> a = torch.tensor([3.0], requires_grad=True)
    >>> y = log_multivariate_gamma(a, 2)
    >>> y.backward()
    >>> a.grad  # sum of digamma(a) + digamma(a - 0.5)
    tensor([1.2740])

    See Also
    --------
    torchscience.special_functions.log_gamma : Logarithm of the gamma function
    torchscience.special_functions.digamma : Digamma function
    scipy.special.multigammaln : SciPy's multivariate log-gamma function
    """
    return torch.ops.torchscience.log_multivariate_gamma(a, d)
