import torch
from torch import Tensor


def zeta(s: Tensor) -> Tensor:
    r"""
    Riemann zeta function.

    Computes the Riemann zeta function evaluated at each element of the input
    tensor. This implementation is restricted to s > 1.

    Mathematical Definition
    -----------------------
    The Riemann zeta function is defined for s > 1 as:

    .. math::

        \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
                 = 1 + \frac{1}{2^s} + \frac{1}{3^s} + \frac{1}{4^s} + \cdots

    Special Values
    --------------
    - zeta(2) = pi^2/6 (Basel problem)
    - zeta(3) = 1.2020569... (Apery's constant)
    - zeta(4) = pi^4/90
    - zeta(s) -> 1 as s -> +infinity
    - zeta(1) = +infinity (pole)

    Domain Restrictions
    -------------------
    - s > 1 : Returns the zeta function value
    - s = 1 : Returns +infinity (pole)
    - s <= 1 : Returns NaN (analytic continuation not implemented)

    Algorithm
    ---------
    Uses Euler-Maclaurin summation with Bernoulli number corrections for
    efficient and accurate computation. The algorithm provides double
    precision accuracy for real inputs.

    Applications
    ------------
    The Riemann zeta function appears in:
    - Number theory (distribution of primes)
    - Physics (quantum field theory, statistical mechanics)
    - Regularization of divergent series
    - Casimir effect calculations
    - Spectral analysis

    Autograd Support
    ----------------
    Gradients are fully supported when s.requires_grad is True.
    The gradient is computed using the derivative formula:

    .. math::

        \frac{d}{ds} \zeta(s) = -\sum_{n=2}^{\infty} \frac{\ln(n)}{n^s}

    Second-order derivatives are also supported.

    Parameters
    ----------
    s : Tensor
        Input tensor. Must have s > 1 for valid results. Can be floating-point
        or complex (with Re(s) > 1).

    Returns
    -------
    Tensor
        The Riemann zeta function evaluated at each element of s. Returns NaN
        for s <= 1 (except s = 1 which returns +infinity).

    Examples
    --------
    Evaluate at positive integers:

    >>> import torch
    >>> import math
    >>> s = torch.tensor([2.0, 3.0, 4.0, 10.0])
    >>> zeta(s)
    tensor([1.6449, 1.2021, 1.0823, 1.0010])

    Verify the Basel problem zeta(2) = pi^2/6:

    >>> s = torch.tensor([2.0], dtype=torch.float64)
    >>> result = zeta(s)
    >>> expected = math.pi**2 / 6
    >>> torch.isclose(result, torch.tensor([expected]))
    tensor([True])

    Check that zeta(s) -> 1 for large s:

    >>> s = torch.tensor([20.0, 50.0])
    >>> zeta(s)
    tensor([1.0000, 1.0000])

    Pole at s = 1:

    >>> s = torch.tensor([1.0])
    >>> zeta(s)
    tensor([inf])

    Autograd:

    >>> s = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    >>> y = zeta(s)
    >>> y.backward()
    >>> s.grad  # derivative of zeta at s=2
    tensor([-0.9376])

    Notes
    -----
    This implementation does NOT support the analytic continuation of the zeta
    function to s <= 1. For s < 1, the function returns NaN. A full
    implementation would use the reflection formula:

    .. math::

        \zeta(s) = 2^s \pi^{s-1} \sin\left(\frac{\pi s}{2}\right)
                   \Gamma(1-s) \zeta(1-s)

    to extend the domain to all complex s except s = 1.

    See Also
    --------
    scipy.special.zeta : SciPy's zeta function (supports full analytic continuation)
    """
    return torch.ops.torchscience.zeta(s)
