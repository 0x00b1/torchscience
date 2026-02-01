import torch
from torch import Tensor


def gegenbauer_polynomial_c(n: Tensor, lambda_: Tensor, z: Tensor) -> Tensor:
    r"""
    Gegenbauer (ultraspherical) polynomial of degree n and parameter lambda.

    Computes the Gegenbauer polynomial :math:`C_n^\lambda(z)` for arbitrary
    complex order n, parameter lambda, and argument z.

    Mathematical Definition
    -----------------------
    For complex n, lambda, and z, uses the hypergeometric representation:

    .. math::

        C_n^\lambda(z) = \frac{\Gamma(n+2\lambda)}{\Gamma(2\lambda)\Gamma(n+1)}
        \cdot {}_2F_1\left(-n, n+2\lambda; \lambda+\frac{1}{2}; \frac{1-z}{2}\right)

    For non-negative integer n, this reduces to the classical Gegenbauer
    polynomial satisfying the recurrence:

    .. math::

        n C_n^\lambda(z) = 2(n+\lambda-1)z C_{n-1}^\lambda(z) - (n+2\lambda-2) C_{n-2}^\lambda(z)

    Special Values
    --------------
    - :math:`C_0^\lambda(z) = 1`
    - :math:`C_1^\lambda(z) = 2\lambda z`
    - :math:`C_n^{1/2}(z) = P_n(z)` (Legendre polynomial)
    - :math:`C_n^1(z) = U_n(z)` (Chebyshev polynomial of the second kind)

    Special Cases
    -------------
    - When :math:`\lambda = 1/2`, Gegenbauer polynomials reduce to Legendre polynomials
    - When :math:`\lambda = 1`, they become Chebyshev polynomials of the second kind
    - When :math:`\lambda = 0`, they are related to Chebyshev polynomials of the first kind

    Domain
    ------
    - n: any real or complex value
    - lambda: any real or complex value (lambda > -1/2 for orthogonality)
    - z: any real or complex value
    - For real z in [-1, 1], values are real for real n and lambda

    Applications
    ------------
    - Angular momentum coupling coefficients in quantum mechanics
    - Expansions on the d-dimensional sphere
    - Orthogonal polynomials with weight function :math:`(1-z^2)^{\lambda-1/2}` on [-1, 1]
    - Spectral methods for solving differential equations

    Dtype Promotion
    ---------------
    - Supports float32, float64, complex64, complex128
    - If any input is complex, output is complex

    Autograd Support
    ----------------
    First and second-order derivatives are supported for n, lambda, and z.

    .. warning::
        Second-order derivatives with respect to n and lambda use finite differences
        and may be less accurate than analytical derivatives.

    Parameters
    ----------
    n : Tensor
        Order of the polynomial. Can be any real or complex value.
    lambda_ : Tensor
        Parameter of the polynomial. Denoted lambda in the mathematical
        definition. Can be any real or complex value.
    z : Tensor
        Argument. Broadcasting with n and lambda is supported.

    Returns
    -------
    Tensor
        The Gegenbauer polynomial :math:`C_n^\lambda(z)`.

    Examples
    --------
    Integer orders:

    >>> n = torch.tensor([0, 1, 2, 3], dtype=torch.float64)
    >>> lam = torch.tensor([1.0], dtype=torch.float64)
    >>> z = torch.tensor([0.5], dtype=torch.float64)
    >>> gegenbauer_polynomial_c(n, lam, z)
    tensor([ 1.0000,  1.0000, -0.5000, -1.0000], dtype=torch.float64)

    Relation to Legendre polynomial (lambda = 0.5):

    >>> n = torch.tensor([2.0], dtype=torch.float64)
    >>> lam = torch.tensor([0.5], dtype=torch.float64)
    >>> z = torch.tensor([0.5], dtype=torch.float64)
    >>> gegenbauer_polynomial_c(n, lam, z)  # Should equal P_2(0.5) = -0.125
    tensor([-0.1250], dtype=torch.float64)

    See Also
    --------
    legendre_polynomial_p : Legendre polynomial (special case of Gegenbauer)
    chebyshev_polynomial_t : Chebyshev polynomial of the first kind
    jacobi_polynomial_p : Generalization of Gegenbauer polynomials
    """
    return torch.ops.torchscience.gegenbauer_polynomial_c(n, lambda_, z)
