import torch
from torch import Tensor


def charlier_polynomial_c(n: Tensor, x: Tensor, a: Tensor) -> Tensor:
    r"""
    Charlier polynomial.

    Computes the Charlier polynomial :math:`C_n(x; a)`.

    Mathematical Definition
    -----------------------
    The Charlier polynomials are discrete orthogonal polynomials associated with
    the Poisson distribution. They satisfy the three-term recurrence relation:

    .. math::

       C_0(x; a) &= 1 \\
       C_1(x; a) &= \frac{x - a}{a} = \frac{x}{a} - 1 \\
       a \, C_{n+1}(x; a) &= (x - n - a) \, C_n(x; a) - n \, C_{n-1}(x; a)

    Alternatively, they can be defined via the hypergeometric representation:

    .. math::

       C_n(x; a) = (-1)^n \, {}_2F_0\left(-n, -x; \,; -\frac{1}{a}\right)

    Special Values
    --------------
    - :math:`C_0(x; a) = 1`
    - :math:`C_1(x; a) = \frac{x}{a} - 1`
    - :math:`C_2(x; a) = \frac{x^2 - (2a+1)x + a^2}{a^2}`

    Orthogonality
    -------------
    The Charlier polynomials are orthogonal with respect to the Poisson weight:

    .. math::

       \sum_{x=0}^{\infty} \frac{a^x}{x!} C_m(x; a) C_n(x; a)
       = e^a \frac{n!}{a^n} \delta_{mn}

    Applications
    ------------
    - **Poisson distribution**: The Charlier polynomials are the natural
      orthogonal polynomials for the Poisson probability mass function.
    - **Combinatorics**: Used in counting problems related to permutations
      and derangements.
    - **Quantum optics**: Appear in the analysis of coherent states.
    - **Signal processing**: Used in discrete signal analysis.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Should be a non-negative integer for the
        polynomial interpretation, though the function is defined for general n.
        Broadcasting with x and a is supported.
    x : Tensor
        Argument at which to evaluate the polynomial.
        Broadcasting with n and a is supported.
    a : Tensor
        Parameter of the Charlier polynomial. Must satisfy :math:`a > 0` for
        the orthogonality relation to hold.
        Broadcasting with n and x is supported.

    Returns
    -------
    Tensor
        The Charlier polynomial :math:`C_n(x; a)` evaluated at the input values.

    Examples
    --------
    Basic usage with small degrees:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> x = torch.tensor([2.0])
    >>> a = torch.tensor([1.0])
    >>> charlier_polynomial_c(n, x, a)
    tensor([1., 1., 0.])

    Verify C_0(x; a) = 1:

    >>> n = torch.tensor([0.0])
    >>> x = torch.tensor([0.0, 1.0, 5.0])
    >>> a = torch.tensor([2.0])
    >>> charlier_polynomial_c(n, x, a)
    tensor([1., 1., 1.])

    Verify C_1(x; a) = x/a - 1:

    >>> n = torch.tensor([1.0])
    >>> x = torch.tensor([0.0, 2.0, 4.0])
    >>> a = torch.tensor([2.0])
    >>> charlier_polynomial_c(n, x, a)
    tensor([-1.,  0.,  1.])

    .. warning:: Gradients use finite differences

       All gradients (with respect to n, x, and a) are computed using
       finite differences and may have reduced accuracy compared to
       analytical gradients.

    Notes
    -----
    - The implementation uses the recurrence relation for non-negative integer n.
    - For non-integer n, linear interpolation between adjacent integer values
      is used.
    - The parameter a must be positive for numerical stability.

    See Also
    --------
    krawtchouk_polynomial_k : Krawtchouk polynomial (discrete orthogonal)
    meixner_polynomial_m : Meixner polynomial (discrete orthogonal)
    laguerre_polynomial_l : Generalized Laguerre polynomial
    """
    return torch.ops.torchscience.charlier_polynomial_c(n, x, a)
