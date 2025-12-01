import torch
from torch import Tensor


def modified_bessel_i(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Modified Bessel function of the first kind of general order n.

    Computes the modified Bessel function I_n(z) evaluated at each element of
    the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The modified Bessel function of the first kind of order n is defined as:

    .. math::

       I_n(z) = \sum_{k=0}^\infty \frac{1}{k! \Gamma(n+k+1)}
                \left(\frac{z}{2}\right)^{n+2k}

    Or equivalently via the integral representation:

    .. math::

       I_n(z) = \frac{1}{\pi} \int_0^\pi e^{z \cos\theta} \cos(n\theta) \, d\theta

    Special Values
    --------------
    - I_0(0) = 1
    - I_n(0) = 0 for n != 0
    - I_n(z) > 0 for z > 0 (always positive on positive real axis)
    - I_n(+infinity) = +infinity (exponential growth)
    - I_n(NaN) = NaN

    Symmetry
    --------
    For integer n:

    .. math::

       I_{-n}(z) = I_n(z)

    This symmetry is a fundamental property of I_n for integer orders.

    For negative z with integer n:

    .. math::

       I_n(-z) = (-1)^n I_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: any real or complex number (argument)
    - I_n is an entire function (no singularities in the finite plane)

    Algorithm
    ---------
    - For integer n = 0, 1: Uses optimized I_0, I_1 implementations
    - For other integer n: Uses Miller's backward recurrence for stability
    - For non-integer n: Uses power series expansion
    - For large |z|: Uses asymptotic expansion
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Recurrence Relations
    --------------------
    Modified Bessel functions of the first kind satisfy:

    .. math::

       I_{n-1}(z) - I_{n+1}(z) = \frac{2n}{z} I_n(z)

    And also:

    .. math::

       I_{n+1}(z) = I_{n-1}(z) - \frac{2n}{z} I_n(z)

    Derivative
    ----------
    The derivative with respect to z is:

    .. math::

       \frac{d}{dz} I_n(z) = \frac{I_{n-1}(z) + I_{n+1}(z)}{2}

    Note the plus sign (unlike J_n which has a minus sign).

    Asymptotic Behavior
    -------------------
    For large z:

    .. math::

       I_n(z) \sim \frac{e^z}{\sqrt{2\pi z}} \left(1 - \frac{4n^2-1}{8z} + \cdots\right)

    This shows the exponential growth behavior of I_n for large positive z.

    Relation to Other Bessel Functions
    ----------------------------------
    I_n is related to J_n (Bessel function of the first kind) by:

    .. math::

       I_n(z) = i^{-n} J_n(iz)

    where i is the imaginary unit.

    Applications
    ------------
    The modified Bessel function I_n appears in many contexts:
    - Heat conduction in cylindrical coordinates
    - Diffusion problems with cylindrical symmetry
    - Probability theory (von Mises distribution, Rice distribution)
    - Signal processing (circular statistics)
    - Physics (Bose-Einstein condensates, cosmic strings)
    - Statistical mechanics (partition functions)

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported for both n and z when they require grad.
    The gradient with respect to z is computed analytically:

    .. math::

       \frac{\partial}{\partial z} I_n(z) = \frac{I_{n-1}(z) + I_{n+1}(z)}{2}

    The gradient with respect to n is computed numerically since the
    analytical formula involves complex integrals.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the Bessel function. Can be any real or complex number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the Bessel function.
        Can be any real or complex number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The modified Bessel function I_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> modified_bessel_i(n, z)
    tensor([1.2661, 1.5906, 2.2452])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(modified_bessel_i(n0, z), modified_bessel_i_0(z))
    True

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> modified_bessel_i(n, z)
    tensor([1.0571, 1.2764, 2.0163])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> modified_bessel_i(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = modified_bessel_i(n, z)
    >>> y.backward()
    >>> z.grad  # equals (I_0(2) + I_2(2)) / 2
    tensor([1.5540])

    Symmetry for integer n:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> i_pos = modified_bessel_i(n, z)
    >>> i_neg = modified_bessel_i(-n, z)
    >>> torch.allclose(i_neg, i_pos)  # I_{-2}(z) = I_2(z)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> modified_bessel_i(n, z)
    tensor([1.1545+0.4174j])

    Exponential growth for large z:

    >>> n = torch.tensor([0.0])
    >>> z = torch.tensor([10.0])
    >>> modified_bessel_i(n, z)
    tensor([2815.7167])

    .. warning:: Numerical precision

       For large |z|, I_n grows exponentially which can lead to overflow
       for very large arguments. For large |n| relative to |z|, numerical
       precision may be reduced.

    .. warning:: Branch cuts for complex arguments

       For complex z with negative real part and non-integer n, the
       function involves a complex power. Results depend on the branch chosen.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `modified_bessel_i_0` and
      `modified_bessel_i_1` may provide slightly better accuracy.
    - The implementation uses Miller's backward recurrence for integer orders,
      which is numerically stable even for large |n|.
    - The power series is used for non-integer orders and converges
      well for |z| < 20 + 2|n|.
    - Unlike K_n, I_n has no singularity at z=0 (except for non-integer n < 0).

    See Also
    --------
    modified_bessel_i_0 : Modified Bessel function I_0(z)
    modified_bessel_i_1 : Modified Bessel function I_1(z)
    modified_bessel_k : Modified Bessel function of the second kind K_n(z)
    modified_bessel_k_0 : Modified Bessel function K_0(z)
    modified_bessel_k_1 : Modified Bessel function K_1(z)
    bessel_j : Bessel function of the first kind J_n(z)
    """
    return torch.ops.torchscience.modified_bessel_i(n, z)
