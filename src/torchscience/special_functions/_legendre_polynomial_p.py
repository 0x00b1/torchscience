import torch
from torch import Tensor


def legendre_polynomial_p(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Legendre polynomial of the first kind.

    Computes the Legendre polynomial P_n(z) for degree n and argument z.

    Mathematical Definition
    -----------------------
    The Legendre polynomial is defined via the Gauss hypergeometric function:

    .. math::

       P_n(z) = {}_2F_1(-n, n+1; 1; (1-z)/2)

    For non-negative integer n, this reduces to a polynomial of degree n.
    Equivalently, using Rodrigues' formula for integer n >= 0:

    .. math::

       P_n(z) = \frac{1}{2^n n!} \frac{d^n}{dz^n}(z^2 - 1)^n

    The Bonnet recurrence relation (for integer n):

    .. math::

       (n+1) P_{n+1}(z) = (2n+1) z P_n(z) - n P_{n-1}(z)

    Special Values
    --------------
    - P_0(z) = 1 for all z
    - P_1(z) = z for all z
    - P_2(z) = (3z^2 - 1)/2
    - P_3(z) = (5z^3 - 3z)/2
    - P_n(1) = 1 for all n >= 0
    - P_n(-1) = (-1)^n for integer n >= 0
    - P_n(0) = 0 for odd n; non-zero for even n

    Domain
    ------
    - n: any real or complex value (integer values are the classical case)
    - z: any real or complex value
    - For real z in [-1, 1]: standard orthogonal polynomial domain
    - For |z| > 1: analytic continuation via hypergeometric

    Algorithm
    ---------
    Uses the hypergeometric representation 2F1(-n, n+1; 1; (1-z)/2) which:
    - Provides a unified implementation for all n (integer and non-integer)
    - Supports complex arguments naturally
    - Has good numerical properties for moderate |z|

    For large integer n or |z| near 1, alternative methods (recurrence,
    asymptotic expansions) may be more accurate.

    Applications
    ------------
    Legendre polynomials are fundamental in mathematical physics:
    - Electrostatics: multipole expansions of charge distributions
    - Gravitational potential: spherical harmonics Y_l^m include P_l^m
    - Quantum mechanics: angular momentum eigenfunctions
    - Numerical integration: Gauss-Legendre quadrature nodes are zeros of P_n
    - Heat conduction: solutions in spherical coordinates
    - Antenna theory: radiation patterns
    - Geophysics: Earth's gravitational and magnetic field models

    Dtype Promotion
    ---------------
    - If either n or z is complex -> output is complex.
    - complex64 if all inputs <= float32/complex64, else complex128.
    - If both real -> standard PyTorch promotion rules apply.
    - Supports float16, bfloat16, float32, float64, complex64, complex128.

    Integer Dtype Handling
    ----------------------
    If n is passed as an integer dtype tensor (e.g., torch.int32, torch.int64),
    it will be promoted to a floating-point dtype via PyTorch's standard type
    promotion rules before computation.

    Autograd Support
    ----------------
    - Gradients for z are computed using the hypergeometric derivative formula.
    - Gradients for n are computed via finite differences (since dP/dn involves
      derivatives of 2F1 with respect to parameters).
    - Second-order derivatives (gradgradcheck) are supported for z.

    .. warning::

       Second-order derivatives with respect to n are approximate and return zero.
       This is because the mixed partial d^2P/dndz and second partial d^2P/dn^2
       involve complex parameter derivatives of the hypergeometric function that
       are not currently implemented. Use with caution when computing Hessians
       involving the degree parameter n.

    Backward formulas:

    .. math::

       \frac{\partial P_n}{\partial z} &= \frac{n(n+1)}{2} \cdot
           {}_2F_1(1-n, n+2; 2; (1-z)/2)

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Can be integer or non-integer.
        For the classical Legendre polynomials, use non-negative integers.
    z : Tensor
        Input tensor. Can be floating-point or complex.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Legendre polynomial P_n(z) evaluated at the input values.
        Output dtype follows the promotion rules described above.

    Examples
    --------
    Integer degree with real input:

    >>> n = torch.tensor([0, 1, 2, 3])
    >>> z = torch.tensor([0.5])
    >>> legendre_polynomial_p(n, z)
    tensor([ 1.0000,  0.5000, -0.1250, -0.4375])

    Verify P_2(z) = (3z^2 - 1)/2:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([0.5])
    >>> legendre_polynomial_p(n, z)
    tensor([-0.1250])
    >>> (3 * 0.5**2 - 1) / 2
    -0.125

    Non-integer degree (generalized Legendre function):

    >>> n = torch.tensor([0.5])
    >>> z = torch.tensor([0.0, 0.5, 1.0])
    >>> legendre_polynomial_p(n, z)
    tensor([0.8409, 0.9213, 1.0000])

    Autograd example:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> y = legendre_polynomial_p(n, z)
    >>> y.backward()
    >>> z.grad  # dP_2/dz = 3z
    tensor([1.5000])

    .. warning:: Numerical accuracy

       For large |n| or |z| near the branch point at z=1, the hypergeometric
       series may converge slowly. Consider using scipy.special.legendre
       for high-precision needs with large parameters.

    .. warning:: Branch cuts

       For complex z, the function has a branch cut from -infinity to -1
       on the real axis. Results may be discontinuous across this cut.

    Notes
    -----
    - The implementation uses 2F1(-n, n+1; 1; (1-z)/2) which is well-defined
      for all n and z.
    - For integer n >= 0, this gives the classical Legendre polynomial.
    - For non-integer n, this is the Legendre function of the first kind.

    See Also
    --------
    scipy.special.eval_legendre : SciPy's Legendre polynomial
    associated_legendre_polynomial_p : Associated Legendre polynomial P_n^m(z)
    """
    return torch.ops.torchscience.legendre_polynomial_p(n, z)
