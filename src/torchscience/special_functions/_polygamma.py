import torch
from torch import Tensor


def polygamma(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Polygamma function.

    Computes the polygamma function of order n evaluated at each element of z.
    The polygamma function is the (n+1)th derivative of the logarithm of the
    gamma function.

    Mathematical Definition
    -----------------------
    The polygamma function is defined as:

    .. math::

       \psi^{(n)}(z) = \frac{d^{n+1}}{dz^{n+1}} \ln \Gamma(z) = \frac{d^n}{dz^n} \psi(z)

    where :math:`\psi(z)` is the digamma function.

    Special Cases
    -------------
    - psi^(0)(z) = digamma(z) = d/dz ln(Gamma(z))
    - psi^(1)(z) = trigamma(z)
    - psi^(2)(z) = tetragamma(z)
    - psi^(3)(z) = pentagamma(z)

    For positive integers, there is a closed form:

    .. math::

       \psi^{(n)}(1) = (-1)^{n+1} n! \zeta(n+1)

    where zeta is the Riemann zeta function.

    Domain
    ------
    - n: non-negative integer (as a tensor)
    - z: any real or complex value except non-positive integers
    - Poles at z = 0, -1, -2, -3, ... for all n >= 0

    Algorithm
    ---------
    - For n = 0, 1, 2, 3: uses optimized implementations (digamma, trigamma,
      tetragamma, pentagamma)
    - For n >= 4: uses recurrence relation and asymptotic expansion
    - Uses reflection formula for Re(z) < 0.5 (complex)

    Recurrence Relations
    --------------------
    - psi^(n)(z+1) = psi^(n)(z) + (-1)^n * n! / z^(n+1)

    Applications
    ------------
    The polygamma function appears in:
    - Higher-order derivatives of the gamma function
    - Statistical mechanics and quantum field theory
    - Computation of moments of distributions
    - Series expansions and asymptotic analysis

    Autograd Support
    ----------------
    Gradients with respect to z are fully supported when z.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dz} \psi^{(n)}(z) = \psi^{(n+1)}(z)

    Gradients with respect to n are not provided since n must be an integer.
    Second-order derivatives are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the polygamma function. Must contain non-negative integers.
        Can be broadcast with z.
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The polygamma function of order n evaluated at each element of z.

    Examples
    --------
    Compute the trigamma function (n=1):

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> polygamma(n, z)
    tensor([1.6449, 0.6449, 0.3949])

    Verify psi^(1)(1) = pi^2/6:

    >>> import math
    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([1.0])
    >>> polygamma(n, z)
    tensor([1.6449])  # approximately pi^2/6 = 1.6449...

    Compare with digamma (n=0):

    >>> n = torch.tensor([0.0])
    >>> z = torch.tensor([2.0])
    >>> polygamma(n, z)
    tensor([0.4228])  # same as digamma(2)

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = polygamma(n, z)
    >>> y.backward()
    >>> z.grad  # psi^(2)(2) = tetragamma(2)
    tensor([-0.4041])

    See Also
    --------
    torchscience.special_functions.digamma : Digamma function (psi^(0))
    torchscience.special_functions.trigamma : Trigamma function (psi^(1))
    torch.special.polygamma : PyTorch's polygamma implementation
    """
    return torch.ops.torchscience.polygamma(n, z)
