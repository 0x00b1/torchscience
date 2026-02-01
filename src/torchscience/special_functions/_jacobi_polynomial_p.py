import torch
from torch import Tensor


def jacobi_polynomial_p(
    n: Tensor, alpha: Tensor, beta: Tensor, z: Tensor
) -> Tensor:
    r"""
    Jacobi polynomial of degree n with parameters alpha and beta.

    Computes the Jacobi polynomial :math:`P_n^{(\alpha,\beta)}(z)` for arbitrary
    complex order n, parameters alpha and beta, and argument z.

    Mathematical Definition
    -----------------------
    For complex n, alpha, beta, and z, uses the hypergeometric representation:

    .. math::

        P_n^{(\alpha,\beta)}(z) = \frac{\Gamma(n+\alpha+1)}{\Gamma(\alpha+1)\Gamma(n+1)}
        \cdot {}_2F_1\left(-n, n+\alpha+\beta+1; \alpha+1; \frac{1-z}{2}\right)

    For non-negative integer n, this is the classical Jacobi polynomial satisfying
    the three-term recurrence relation.

    Special Values
    --------------
    - :math:`P_0^{(\alpha,\beta)}(z) = 1`
    - :math:`P_1^{(\alpha,\beta)}(z) = \frac{\alpha-\beta}{2} + \frac{\alpha+\beta+2}{2}z`

    Special Cases
    -------------
    - :math:`P_n^{(0,0)}(z) = P_n(z)` (Legendre polynomial)
    - :math:`P_n^{(-1/2,-1/2)}(z) = \frac{(2n)!}{4^n (n!)^2} T_n(z)` (related to Chebyshev T)
    - :math:`P_n^{(1/2,1/2)}(z) = \frac{(2n+1)!}{4^n (n!)^2 (n+1)} U_n(z)` (related to Chebyshev U)
    - When :math:`\alpha = \beta = \lambda - 1/2`, proportional to Gegenbauer :math:`C_n^\lambda(z)`

    Domain
    ------
    - n: any real or complex value
    - alpha: any real or complex value (alpha > -1 for orthogonality)
    - beta: any real or complex value (beta > -1 for orthogonality)
    - z: any real or complex value
    - For real z in [-1, 1], values are real for real n, alpha, and beta

    Applications
    ------------
    - Most general classical orthogonal polynomial family on [-1, 1]
    - Weight function :math:`(1-z)^\alpha (1+z)^\beta` on [-1, 1]
    - Spectral methods for solving differential equations
    - Quadrature rules (Gauss-Jacobi quadrature)
    - Angular momentum in quantum mechanics

    Dtype Promotion
    ---------------
    - Supports float32, float64, complex64, complex128
    - If any input is complex, output is complex

    Autograd Support
    ----------------
    First and second-order derivatives are supported for n, alpha, beta, and z.

    .. warning::
        Second-order derivatives with respect to n, alpha, and beta use finite
        differences and may be less accurate than analytical derivatives.

    Parameters
    ----------
    n : Tensor
        Order of the polynomial. Can be any real or complex value.
    alpha : Tensor
        First parameter of the polynomial. Can be any real or complex value.
    beta : Tensor
        Second parameter of the polynomial. Can be any real or complex value.
    z : Tensor
        Argument. Broadcasting with n, alpha, and beta is supported.

    Returns
    -------
    Tensor
        The Jacobi polynomial :math:`P_n^{(\alpha,\beta)}(z)`.

    Examples
    --------
    Integer orders with alpha=beta=0 (Legendre case):

    >>> n = torch.tensor([0, 1, 2, 3], dtype=torch.float64)
    >>> alpha = torch.tensor([0.0], dtype=torch.float64)
    >>> beta = torch.tensor([0.0], dtype=torch.float64)
    >>> z = torch.tensor([0.5], dtype=torch.float64)
    >>> jacobi_polynomial_p(n, alpha, beta, z)
    tensor([ 1.0000,  0.5000, -0.1250, -0.4375], dtype=torch.float64)

    General parameters:

    >>> n = torch.tensor([2.0], dtype=torch.float64)
    >>> alpha = torch.tensor([0.5], dtype=torch.float64)
    >>> beta = torch.tensor([1.0], dtype=torch.float64)
    >>> z = torch.tensor([0.5], dtype=torch.float64)
    >>> jacobi_polynomial_p(n, alpha, beta, z)
    tensor([...], dtype=torch.float64)

    See Also
    --------
    legendre_polynomial_p : Legendre polynomial (special case with alpha=beta=0)
    gegenbauer_polynomial_c : Gegenbauer polynomial (special case with alpha=beta)
    chebyshev_polynomial_t : Chebyshev polynomial of the first kind
    """
    return torch.ops.torchscience.jacobi_polynomial_p(n, alpha, beta, z)
