import torch
from torch import Tensor


def hahn_polynomial_q(
    n: Tensor, x: Tensor, alpha: Tensor, beta: Tensor, N: Tensor
) -> Tensor:
    r"""
    Hahn polynomial of degree n.

    Computes the Hahn polynomial :math:`Q_n(x; \alpha, \beta, N)` for given degree n,
    evaluation point x, parameters alpha and beta, and size parameter N.

    Mathematical Definition
    -----------------------
    Via generalized hypergeometric function:

    .. math::

        Q_n(x; \alpha, \beta, N) = {}_3F_2(-n, n+\alpha+\beta+1, -x; \alpha+1, -N; 1)

    Via explicit summation:

    .. math::

        Q_n(x; \alpha, \beta, N) = \sum_{k=0}^{n}
            \frac{(-n)_k (n+\alpha+\beta+1)_k (-x)_k}{(\alpha+1)_k (-N)_k k!}

    where :math:`(a)_k` is the Pochhammer symbol (rising factorial).

    Special Values
    --------------
    - :math:`Q_0(x; \alpha, \beta, N) = 1` for all valid parameters
    - :math:`Q_1(x; \alpha, \beta, N) = 1 - \frac{(\alpha+\beta+2) x}{(\alpha+1) N}`

    Domain
    ------
    - n: degree, typically non-negative integer with 0 <= n <= N
    - x: evaluation point, typically integer with 0 <= x <= N
    - alpha: parameter, alpha > -1
    - beta: parameter, beta > -1
    - N: size parameter, positive integer

    Orthogonality
    -------------
    The Hahn polynomials are orthogonal with respect to a discrete measure on
    {0, 1, ..., N}:

    .. math::

        \sum_{x=0}^{N} w(x) Q_m(x; \alpha, \beta, N) Q_n(x; \alpha, \beta, N) = h_n \delta_{mn}

    where the weight function involves binomial coefficients with parameters
    alpha and beta.

    Relation to Other Polynomials
    -----------------------------
    The Hahn polynomials generalize several classical orthogonal polynomials:

    - Krawtchouk: special case when alpha = beta and appropriate parameter choices
    - Meixner: limiting case as N -> infinity with appropriate scaling
    - Jacobi: continuous limit

    Applications
    ------------
    - Quantum mechanics (angular momentum coupling)
    - Signal processing
    - Coding theory
    - Approximation theory
    - Stochastic processes

    Dtype Promotion
    ---------------
    - Supports float32, float64, complex64, complex128
    - If any input is complex, output is complex

    Autograd Support
    ----------------
    First and second-order derivatives are supported for n, x, alpha, beta, and N.

    .. warning::
        Derivatives are computed via finite differences and may be less
        accurate than analytical derivatives.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Typically non-negative integer.
    x : Tensor
        Evaluation point. Typically integer in [0, N].
    alpha : Tensor
        First parameter. Must satisfy alpha > -1.
    beta : Tensor
        Second parameter. Must satisfy beta > -1.
    N : Tensor
        Size parameter. Positive integer.

    Returns
    -------
    Tensor
        The Hahn polynomial :math:`Q_n(x; \alpha, \beta, N)`.

    Examples
    --------
    Basic evaluation:

    >>> n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> alpha = torch.tensor([0.5], dtype=torch.float64)
    >>> beta = torch.tensor([0.5], dtype=torch.float64)
    >>> N = torch.tensor([5.0], dtype=torch.float64)
    >>> hahn_polynomial_q(n, x, alpha, beta, N)
    tensor([1.0000, ...], dtype=torch.float64)

    Q_0 is always 1:

    >>> n = torch.tensor([0.0], dtype=torch.float64)
    >>> x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    >>> alpha = torch.tensor([1.0], dtype=torch.float64)
    >>> beta = torch.tensor([1.0], dtype=torch.float64)
    >>> N = torch.tensor([5.0], dtype=torch.float64)
    >>> hahn_polynomial_q(n, x, alpha, beta, N)
    tensor([1., 1., 1., 1.], dtype=torch.float64)

    Q_1 formula:

    >>> n = torch.tensor([1.0], dtype=torch.float64)
    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> alpha = torch.tensor([1.0], dtype=torch.float64)
    >>> beta = torch.tensor([1.0], dtype=torch.float64)
    >>> N = torch.tensor([5.0], dtype=torch.float64)
    >>> # Q_1 = 1 - (alpha+beta+2)*x / ((alpha+1)*N) = 1 - 4*1/(2*5) = 0.6
    >>> hahn_polynomial_q(n, x, alpha, beta, N)
    tensor([0.6000], dtype=torch.float64)

    See Also
    --------
    krawtchouk_polynomial_k : Krawtchouk polynomial (special case of Hahn)
    jacobi_polynomial_p : Jacobi polynomial (continuous analog)
    meixner_polynomial_m : Meixner polynomial (related discrete polynomial)
    """
    return torch.ops.torchscience.hahn_polynomial_q(n, x, alpha, beta, N)
