import torch
from torch import Tensor


def hermite_polynomial_h(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Physicists' Hermite polynomial of degree n.

    Computes the Physicists' Hermite polynomial H_n(z) for degree n and argument z.

    Mathematical Definition
    -----------------------
    The Physicists' Hermite polynomial is defined via the confluent hypergeometric
    function:

    .. math::

       H_n(z) = \frac{2^n \sqrt{\pi}}{\Gamma((1-n)/2)} \cdot
           {}_1F_1\left(-\frac{n}{2}; \frac{1}{2}; z^2\right)
         - \frac{2^n \sqrt{\pi} z}{\Gamma(-n/2)} \cdot
           {}_1F_1\left(\frac{1-n}{2}; \frac{3}{2}; z^2\right)

    For non-negative integer n, this reduces to a polynomial of degree n.
    Equivalently, using Rodrigues' formula for integer n >= 0:

    .. math::

       H_n(z) = (-1)^n e^{z^2} \frac{d^n}{dz^n} e^{-z^2}

    The recurrence relation (for integer n):

    .. math::

       H_{n+1}(z) = 2z H_n(z) - 2n H_{n-1}(z)

    Special Values
    --------------
    - H_0(z) = 1 for all z
    - H_1(z) = 2z for all z
    - H_2(z) = 4z^2 - 2
    - H_3(z) = 8z^3 - 12z
    - H_4(z) = 16z^4 - 48z^2 + 12
    - H_5(z) = 32z^5 - 160z^3 + 120z
    - H_n(0) = 0 for odd n; (-1)^{n/2} * (n-1)!! * 2^{n/2} for even n

    Domain
    ------
    - n: any real or complex value (integer values are the classical case)
    - z: any real or complex value

    Algorithm
    ---------
    Uses the hypergeometric representation which:
    - Provides a unified implementation for all n (integer and non-integer)
    - Supports complex arguments naturally
    - Has good numerical properties for moderate |z|

    For large integer n or |z|, alternative methods (recurrence,
    asymptotic expansions) may be more accurate.

    Applications
    ------------
    Physicists' Hermite polynomials are fundamental in mathematical physics:
    - Quantum mechanics: wave functions of the quantum harmonic oscillator
    - Statistical mechanics: energy levels and partition functions
    - Gaussian quadrature: Gauss-Hermite quadrature nodes and weights
    - Probability: moments of the normal distribution
    - Signal processing: Hermite functions in time-frequency analysis
    - Optics: Hermite-Gaussian beam modes

    Relation to Probabilists' Hermite Polynomials
    ----------------------------------------------
    The probabilists' Hermite polynomials He_n(z) are related by:

    .. math::

       H_n(z) = 2^{n/2} He_n(z \sqrt{2})

       He_n(z) = 2^{-n/2} H_n(z / \sqrt{2})

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
    - Gradients for z are computed using the recurrence dH_n/dz = 2n * H_{n-1}(z).
    - Gradients for n are computed via finite differences (since dH/dn involves
      derivatives of hypergeometric functions with respect to parameters).
    - Second-order derivatives (gradgradcheck) are supported for z.

    .. warning::

       Second-order derivatives with respect to n are approximate and return zero.
       This is because the mixed partial d^2H/dndz and second partial d^2H/dn^2
       involve complex parameter derivatives of the hypergeometric function that
       are not currently implemented. Use with caution when computing Hessians
       involving the degree parameter n.

    Backward formulas:

    .. math::

       \frac{\partial H_n}{\partial z} &= 2n \cdot H_{n-1}(z)

       \frac{\partial^2 H_n}{\partial z^2} &= 4n(n-1) \cdot H_{n-2}(z)

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
        The Physicists' Hermite polynomial H_n(z) evaluated at the input values.
        Output dtype follows the promotion rules described above.

    Examples
    --------
    Integer degree with real input:

    >>> n = torch.tensor([0, 1, 2, 3])
    >>> z = torch.tensor([1.0])
    >>> hermite_polynomial_h(n, z)
    tensor([ 1.,  2.,  2., -4.])

    Verify H_2(z) = 4z^2 - 2:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([0.5])
    >>> hermite_polynomial_h(n, z)
    tensor([-1.])
    >>> 4 * 0.5**2 - 2
    -1.0

    Non-integer degree (generalized Hermite function):

    >>> n = torch.tensor([0.5])
    >>> z = torch.tensor([0.0, 0.5, 1.0])
    >>> hermite_polynomial_h(n, z)  # doctest: +SKIP
    tensor([...])

    Autograd example:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> y = hermite_polynomial_h(n, z)
    >>> y.backward()
    >>> z.grad  # dH_2/dz = 2 * 2 * H_1(z) = 4 * 2z = 8z
    tensor([4.])

    .. warning:: Numerical accuracy

       For large |n| or |z|, the hypergeometric series may converge slowly.
       Consider using scipy.special.eval_hermite for high-precision needs
       with large parameters.

    Notes
    -----
    - The implementation uses the confluent hypergeometric representation which
      is well-defined for all n and z.
    - For integer n >= 0, this gives the classical Physicists' Hermite polynomial.
    - For non-integer n, this is a generalized Hermite function.

    See Also
    --------
    scipy.special.eval_hermite : SciPy's Physicists' Hermite polynomial
    hermite_polynomial_he : Probabilists' Hermite polynomial He_n(z)
    """
    return torch.ops.torchscience.hermite_polynomial_h(n, z)
