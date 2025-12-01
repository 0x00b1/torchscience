import torch
from torch import Tensor


def binomial_coefficient(n: Tensor, k: Tensor) -> Tensor:
    r"""
    Binomial coefficient.

    Computes the generalized binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    using the gamma function representation for numerical stability and to
    support non-integer arguments.

    Mathematical Definition
    -----------------------
    The binomial coefficient is defined as:

    .. math::

       \binom{n}{k} = \frac{\Gamma(n+1)}{\Gamma(k+1) \Gamma(n-k+1)}

    For non-negative integers, this equals the number of ways to choose
    k items from n items without replacement.

    Special Values
    --------------
    - C(n, 0) = 1 for all n
    - C(n, n) = 1 for non-negative integer n
    - C(n, k) = 0 for k < 0 or (n >= 0 and k > n)
    - C(n, k) = C(n, n-k) (symmetry for non-negative integer n)

    Generalized Binomial Coefficients
    ---------------------------------
    For non-integer n, this computes the generalized binomial coefficient:

    .. math::

       \binom{n}{k} = \frac{n(n-1)(n-2)\cdots(n-k+1)}{k!}

    This extends the binomial coefficient to negative and fractional n,
    which appears in the binomial series expansion of (1+x)^n.

    Autograd Support
    ----------------
    Gradients are fully supported when n.requires_grad or k.requires_grad
    is True. The gradients are computed using the digamma function:

    .. math::

       \frac{\partial}{\partial n} \binom{n}{k} = \binom{n}{k}
           \left( \psi(n+1) - \psi(n-k+1) \right)

    .. math::

       \frac{\partial}{\partial k} \binom{n}{k} = \binom{n}{k}
           \left( -\psi(k+1) + \psi(n-k+1) \right)

    Second-order derivatives are also supported using the trigamma function.

    Parameters
    ----------
    n : Tensor
        Number of items to choose from. Can be any real number for
        generalized binomial coefficients.
    k : Tensor
        Number of items to choose. Must be broadcastable with n.

    Returns
    -------
    Tensor
        The binomial coefficient C(n, k) for each element pair.
        Output shape is the broadcast shape of n and k.

    Examples
    --------
    Integer binomial coefficients (Pascal's triangle):

    >>> n = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    >>> k = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    >>> binomial_coefficient(n, k)
    tensor([ 1.,  5., 10., 10.,  5.,  1.])

    Generalized binomial coefficient with negative n:

    >>> n = torch.tensor([-0.5])
    >>> k = torch.tensor([2.0])
    >>> binomial_coefficient(n, k)
    tensor([0.375])  # (-0.5)(-1.5) / 2! = 0.75 / 2 = 0.375

    Autograd example:

    >>> n = torch.tensor([5.0], requires_grad=True)
    >>> k = torch.tensor([2.0])
    >>> result = binomial_coefficient(n, k)
    >>> result.backward()
    >>> n.grad
    tensor([...])  # gradient w.r.t. n

    See Also
    --------
    torchscience.special_functions.gamma : Gamma function
    torchscience.special_functions.log_gamma : Log-gamma function
    scipy.special.comb : SciPy's combination function
    """
    return torch.ops.torchscience.binomial_coefficient(n, k)
