import torch
from torch import Tensor


def meixner_polynomial_m(
    n: Tensor, x: Tensor, beta: Tensor, c: Tensor
) -> Tensor:
    r"""
    Meixner polynomial of degree n.

    Computes the Meixner polynomial :math:`M_n(x; \beta, c)` for given degree n,
    evaluation point x, parameter beta, and parameter c.

    Mathematical Definition
    -----------------------
    Via hypergeometric function:

    .. math::

        M_n(x; \beta, c) = {}_2F_1(-n, -x; \beta; 1 - 1/c)

    Via recurrence relation:

    .. math::

        M_0(x; \beta, c) &= 1 \\
        M_1(x; \beta, c) &= 1 + \frac{x(c-1)}{c \beta} \\
        c(n + \beta) M_{n+1}(x) &= [(c-1)x + (1+c)n + c\beta] M_n(x) - n M_{n-1}(x)

    Special Values
    --------------
    - :math:`M_0(x; \beta, c) = 1` for all valid parameters
    - :math:`M_1(x; \beta, c) = 1 + \frac{x(c-1)}{c \beta}`
    - :math:`M_n(0; \beta, c) = 1` for all n

    Domain
    ------
    - n: degree (n >= 0)
    - x: evaluation point
    - beta: parameter (beta > 0)
    - c: parameter (0 < c < 1)

    Orthogonality
    -------------
    The Meixner polynomials are orthogonal with respect to the negative
    binomial distribution on the non-negative integers:

    .. math::

        \sum_{x=0}^{\infty} \frac{(\beta)_x}{x!} c^x M_m(x; \beta, c) M_n(x; \beta, c)
        = \frac{n!}{(\beta)_n c^n (1-c)^\beta} \delta_{mn}

    where :math:`(\beta)_x` is the Pochhammer symbol.

    Applications
    ------------
    - Probability theory (negative binomial distribution)
    - Quantum physics
    - Coding theory
    - Combinatorics
    - Birth-death processes

    Dtype Promotion
    ---------------
    - Supports float32, float64, complex64, complex128
    - If any input is complex, output is complex

    Autograd Support
    ----------------
    First and second-order derivatives are supported for n, x, beta, and c.

    .. warning::
        Derivatives are computed via finite differences and may be less
        accurate than analytical derivatives.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Typically non-negative integer.
    x : Tensor
        Evaluation point.
    beta : Tensor
        Parameter. Must satisfy beta > 0.
    c : Tensor
        Parameter. Must satisfy 0 < c < 1.

    Returns
    -------
    Tensor
        The Meixner polynomial :math:`M_n(x; \beta, c)`.

    Examples
    --------
    Basic evaluation:

    >>> n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> beta = torch.tensor([2.0], dtype=torch.float64)
    >>> c = torch.tensor([0.5], dtype=torch.float64)
    >>> meixner_polynomial_m(n, x, beta, c)
    tensor([1.0000, 0.5000, ...], dtype=torch.float64)

    M_0 is always 1:

    >>> n = torch.tensor([0.0], dtype=torch.float64)
    >>> x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    >>> beta = torch.tensor([2.0], dtype=torch.float64)
    >>> c = torch.tensor([0.5], dtype=torch.float64)
    >>> meixner_polynomial_m(n, x, beta, c)
    tensor([1., 1., 1., 1.], dtype=torch.float64)

    M_1 formula: M_1(x; beta, c) = 1 + x*(c-1)/(c*beta)

    >>> n = torch.tensor([1.0], dtype=torch.float64)
    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> beta = torch.tensor([2.0], dtype=torch.float64)
    >>> c = torch.tensor([0.5], dtype=torch.float64)
    >>> meixner_polynomial_m(n, x, beta, c)  # = 1 + 1*(0.5-1)/(0.5*2) = 0.5
    tensor([0.5000], dtype=torch.float64)

    See Also
    --------
    krawtchouk_polynomial_k : Krawtchouk polynomial
    jacobi_polynomial_p : Jacobi polynomial
    chebyshev_polynomial_t : Chebyshev polynomial of the first kind
    """
    return torch.ops.torchscience.meixner_polynomial_m(n, x, beta, c)
