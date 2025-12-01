import torch
from torch import Tensor


def bessel_j(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Bessel function of the first kind of general order n.

    Computes the Bessel function J_n(z) evaluated at each element of the
    input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The Bessel function of the first kind of order n is defined as:

    .. math::

       J_n(z) = \sum_{k=0}^\infty \frac{(-1)^k}{k! \Gamma(n+k+1)}
                \left(\frac{z}{2}\right)^{n+2k}

    Or equivalently via the integral representation (for integer n):

    .. math::

       J_n(z) = \frac{1}{\pi} \int_0^\pi \cos(n\theta - z\sin\theta) \, d\theta

    Special Values
    --------------
    - J_0(0) = 1
    - J_n(0) = 0 for n > 0
    - J_n(+inf) = 0 for all n
    - J_n(NaN) = NaN

    Symmetry
    --------
    For integer n:

    .. math::

       J_{-n}(z) = (-1)^n J_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: any real or complex number (argument)
    - For non-integer negative n at z=0, the function is singular

    Algorithm
    ---------
    - For integer n = 0, 1: Uses optimized J_0, J_1 implementations
    - For other integer n: Uses Miller's backward recurrence algorithm
    - For non-integer n: Uses power series expansion
    - For large |z|: Uses asymptotic expansion
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Recurrence Relation
    -------------------
    Bessel functions satisfy the recurrence:

    .. math::

       J_{n-1}(z) + J_{n+1}(z) = \frac{2n}{z} J_n(z)

    This can be used to compute J_n for any integer order from J_0 and J_1.

    Applications
    ------------
    The Bessel function J_n appears in many contexts:
    - Vibrations of circular membranes
    - Heat conduction in cylinders
    - Electromagnetic wave propagation in cylindrical coordinates
    - Signal processing (FM synthesis, filter design)
    - Quantum mechanics (hydrogen atom, scattering)

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

       \frac{\partial}{\partial z} J_n(z) = \frac{J_{n-1}(z) - J_{n+1}(z)}{2}

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
        The Bessel function J_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel_j(n, z)
    tensor([0.7652, 0.5767, 0.4861])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(bessel_j(n0, z), bessel_j_0(z))
    True

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel_j(n, z)
    tensor([0.6714, 0.5130, 0.4087])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> bessel_j(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = bessel_j(n, z)
    >>> y.backward()
    >>> z.grad  # equals (J_0(2) - J_2(2)) / 2
    tensor([0.0638])

    Symmetry for integer n:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> j_pos = bessel_j(n, z)
    >>> j_neg = bessel_j(-n, z)
    >>> torch.allclose(j_neg, j_pos)  # J_{-2}(z) = J_2(z)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> bessel_j(n, z)
    tensor([0.8102+0.1122j])

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision due to
       the oscillatory nature of Bessel functions and potential numerical
       cancellation.

    .. warning:: Branch cuts for complex arguments

       For complex z with negative real part and non-integer n, the
       function has a branch cut. Results depend on the branch chosen.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `bessel_j_0` and
      `bessel_j_1` may provide slightly better accuracy.
    - The implementation uses Miller's backward recurrence for integer
      orders, which is numerically stable even for large |n|.
    - The power series is used for non-integer orders and converges
      well for |z| < 20 + 2|n|.

    See Also
    --------
    bessel_j_0 : Bessel function of the first kind of order zero
    bessel_j_1 : Bessel function of the first kind of order one
    bessel_y_0 : Bessel function of the second kind of order zero
    bessel_y_1 : Bessel function of the second kind of order one
    """
    return torch.ops.torchscience.bessel_j(n, z)
