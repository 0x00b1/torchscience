import torch
from torch import Tensor


def krawtchouk_polynomial_k(
    n: Tensor, x: Tensor, p: Tensor, N: Tensor
) -> Tensor:
    r"""
    Krawtchouk polynomial of degree n.

    Computes the Krawtchouk polynomial :math:`K_n(x; p, N)` for given degree n,
    evaluation point x, probability parameter p, and size parameter N.

    Mathematical Definition
    -----------------------
    Via hypergeometric function:

    .. math::

        K_n(x; p, N) = {}_2F_1(-n, -x; -N; 1/p)

    Via recurrence relation:

    .. math::

        K_0(x; p, N) &= 1 \\
        K_1(x; p, N) &= 1 - \frac{x}{Np} \\
        (n+1) K_{n+1}(x) &= [(1-p)(N-n) + pn - x] K_n(x) - p(1-p)(N-n+1) K_{n-1}(x)

    Special Values
    --------------
    - :math:`K_n(0; p, N) = 1` for all n
    - :math:`K_n(N; p, N) = (-1)^n \left(\frac{1-p}{p}\right)^n`

    Domain
    ------
    - n: degree, typically non-negative integer with 0 <= n <= N
    - x: evaluation point, typically integer with 0 <= x <= N
    - p: probability parameter, 0 < p < 1
    - N: size parameter, positive integer

    Orthogonality
    -------------
    The Krawtchouk polynomials are orthogonal with respect to the binomial
    distribution on {0, 1, ..., N}:

    .. math::

        \sum_{x=0}^{N} \binom{N}{x} p^x (1-p)^{N-x} K_m(x; p, N) K_n(x; p, N) = \delta_{mn} h_n

    where :math:`h_n = \frac{(-1)^n n!}{(-N)_n} \left(\frac{1-p}{p}\right)^n`.

    Applications
    ------------
    - Combinatorics and coding theory (MacWilliams transform)
    - Image processing (Krawtchouk moments)
    - Probability theory (binomial distribution analysis)
    - Quantum mechanics

    Dtype Promotion
    ---------------
    - Supports float32, float64, complex64, complex128
    - If any input is complex, output is complex

    Autograd Support
    ----------------
    First and second-order derivatives are supported for n, x, p, and N.

    .. warning::
        Derivatives are computed via finite differences and may be less
        accurate than analytical derivatives.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Typically non-negative integer.
    x : Tensor
        Evaluation point. Typically integer in [0, N].
    p : Tensor
        Probability parameter. Must satisfy 0 < p < 1.
    N : Tensor
        Size parameter. Positive integer.

    Returns
    -------
    Tensor
        The Krawtchouk polynomial :math:`K_n(x; p, N)`.

    Examples
    --------
    Basic evaluation:

    >>> n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> p = torch.tensor([0.5], dtype=torch.float64)
    >>> N = torch.tensor([5.0], dtype=torch.float64)
    >>> krawtchouk_polynomial_k(n, x, p, N)
    tensor([1.0000, 0.6000, ...], dtype=torch.float64)

    K_0 is always 1:

    >>> n = torch.tensor([0.0], dtype=torch.float64)
    >>> x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    >>> p = torch.tensor([0.5], dtype=torch.float64)
    >>> N = torch.tensor([5.0], dtype=torch.float64)
    >>> krawtchouk_polynomial_k(n, x, p, N)
    tensor([1., 1., 1., 1.], dtype=torch.float64)

    K_1 formula:

    >>> n = torch.tensor([1.0], dtype=torch.float64)
    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> p = torch.tensor([0.5], dtype=torch.float64)
    >>> N = torch.tensor([5.0], dtype=torch.float64)
    >>> krawtchouk_polynomial_k(n, x, p, N)  # = 1 - 1/(5*0.5) = 0.6
    tensor([0.6000], dtype=torch.float64)

    See Also
    --------
    jacobi_polynomial_p : Jacobi polynomial
    chebyshev_polynomial_t : Chebyshev polynomial of the first kind
    """
    return torch.ops.torchscience.krawtchouk_polynomial_k(n, x, p, N)
