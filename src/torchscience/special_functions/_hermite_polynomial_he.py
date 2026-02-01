import torch
from torch import Tensor


def hermite_polynomial_he(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Probabilists' Hermite polynomial of degree n.

    Computes the Probabilists' Hermite polynomial He_n(z) for degree n and argument z.

    Mathematical Definition
    -----------------------
    The Probabilists' Hermite polynomial is a simple scaling transformation of the
    Physicists' Hermite polynomial:

    .. math::

       He_n(z) = 2^{-n/2} \cdot H_n\left(\frac{z}{\sqrt{2}}\right)

    Equivalently, using Rodrigues' formula for integer n >= 0:

    .. math::

       He_n(z) = (-1)^n e^{z^2/2} \frac{d^n}{dz^n} e^{-z^2/2}

    The recurrence relation (for integer n):

    .. math::

       He_{n+1}(z) = z \cdot He_n(z) - n \cdot He_{n-1}(z)

    with initial conditions He_0(z) = 1, He_1(z) = z.

    Special Values
    --------------
    - He_0(z) = 1 for all z
    - He_1(z) = z for all z
    - He_2(z) = z^2 - 1
    - He_3(z) = z^3 - 3z
    - He_4(z) = z^4 - 6z^2 + 3
    - He_5(z) = z^5 - 10z^3 + 15z
    - He_n(0) = 0 for odd n; (-1)^{n/2} * (n-1)!! for even n

    Domain
    ------
    - n: any real or complex value (integer values are the classical case)
    - z: any real or complex value

    Relation to Physicists' Hermite Polynomials
    --------------------------------------------
    The Probabilists' and Physicists' Hermite polynomials are related by:

    .. math::

       He_n(z) = 2^{-n/2} H_n(z / \sqrt{2})

       H_n(z) = 2^{n/2} He_n(z \sqrt{2})

    Applications
    ------------
    Probabilists' Hermite polynomials appear naturally in:
    - Probability theory: orthogonal polynomials for standard normal distribution
    - Edgeworth expansion: approximating probability distributions
    - Gaussian integrals: computing moments and integrals
    - Stochastic calculus: Wiener chaos expansion
    - Statistics: cumulants and moments of normal distributions
    - Financial mathematics: option pricing with non-Gaussian returns

    Orthogonality
    -------------
    The Probabilists' Hermite polynomials are orthogonal with respect to the
    standard Gaussian weight function:

    .. math::

       \int_{-\infty}^{\infty} He_m(z) He_n(z) \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz
       = n! \cdot \delta_{mn}

    This differs from the Physicists' version which uses the weight exp(-z^2).

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
    - Gradients for z are computed using the recurrence dHe_n/dz = n * He_{n-1}(z).
    - Gradients for n are computed via finite differences.
    - Second-order derivatives (gradgradcheck) are supported for z.

    .. warning::

       Second-order derivatives with respect to n are approximate and return zero.
       This is because the mixed partial d^2He/dndz and second partial d^2He/dn^2
       involve complex parameter derivatives that are not currently implemented.
       Use with caution when computing Hessians involving the degree parameter n.

    Backward formulas:

    .. math::

       \frac{\partial He_n}{\partial z} &= n \cdot He_{n-1}(z)

       \frac{\partial^2 He_n}{\partial z^2} &= n(n-1) \cdot He_{n-2}(z)

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Can be integer or non-integer.
        For the classical Hermite polynomials, use non-negative integers.
    z : Tensor
        Input tensor. Can be floating-point or complex.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Probabilists' Hermite polynomial He_n(z) evaluated at the input values.
        Output dtype follows the promotion rules described above.

    Examples
    --------
    Integer degree with real input:

    >>> n = torch.tensor([0, 1, 2, 3])
    >>> z = torch.tensor([1.0])
    >>> hermite_polynomial_he(n, z)
    tensor([ 1.,  1.,  0., -2.])

    Verify He_2(z) = z^2 - 1:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([2.0])
    >>> hermite_polynomial_he(n, z)
    tensor([3.])
    >>> 2.0**2 - 1
    3.0

    Relation to H_n:

    >>> import math
    >>> n = torch.tensor([3.0])
    >>> z = torch.tensor([1.5])
    >>> He = hermite_polynomial_he(n, z)
    >>> H = torchscience.special_functions.hermite_polynomial_h(n, z / math.sqrt(2))
    >>> He_from_H = 2**(-3/2) * H
    >>> torch.allclose(He, He_from_H)
    True

    Autograd example:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([1.0], requires_grad=True)
    >>> y = hermite_polynomial_he(n, z)
    >>> y.backward()
    >>> z.grad  # dHe_2/dz = 2 * He_1(z) = 2z = 2
    tensor([2.])

    .. warning:: Numerical accuracy

       For large |n| or |z|, the underlying hypergeometric series may converge
       slowly. Consider using scipy.special.eval_hermitenorm for high-precision
       needs with large parameters.

    Notes
    -----
    - The implementation uses the relation to the Physicists' Hermite polynomial
      via the scaling transformation.
    - For integer n >= 0, this gives the classical Probabilists' Hermite polynomial.
    - For non-integer n, this is a generalized Hermite function.

    See Also
    --------
    scipy.special.eval_hermitenorm : SciPy's Probabilists' Hermite polynomial
    hermite_polynomial_h : Physicists' Hermite polynomial H_n(z)
    """
    return torch.ops.torchscience.hermite_polynomial_he(n, z)
