import torch
from torch import Tensor


def bessel_y(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Bessel function of the second kind of general order n.

    Computes the Bessel function Y_n(z) evaluated at each element of the
    input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    For non-integer order n, the Bessel function of the second kind is defined as:

    .. math::

       Y_n(z) = \frac{J_n(z) \cos(n\pi) - J_{-n}(z)}{\sin(n\pi)}

    where :math:`J_n(z)` is the Bessel function of the first kind.

    For integer order n, this is defined via L'Hopital's rule, involving
    logarithms and digamma functions:

    .. math::

       Y_n(z) = \frac{2}{\pi}\left[J_n(z)\ln\frac{z}{2} - \sum_{k=0}^{n-1}
                \frac{(n-k-1)!}{k!}\left(\frac{z}{2}\right)^{2k-n}
                - \sum_{k=0}^\infty \frac{(-1)^k(\psi(k+1)+\psi(n+k+1))}{k!(n+k)!}
                \left(\frac{z}{2}\right)^{n+2k}\right]

    Special Values
    --------------
    - Y_n(0) = -inf for all n (singularity at origin)
    - Y_n(+inf) = 0 for all n
    - Y_n(NaN) = NaN
    - Y_n(z) for z < 0 (real) = NaN (branch cut along negative real axis)

    Symmetry
    --------
    For integer n:

    .. math::

       Y_{-n}(z) = (-1)^n Y_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: positive real numbers, or complex numbers (not on negative real axis)
    - For z <= 0 with real z, the function returns -inf at z=0 and NaN for z<0

    Algorithm
    ---------
    - For integer n = 0, 1: Uses optimized Y_0, Y_1 implementations (Cephes)
    - For other integer n: Uses forward recurrence from Y_0 and Y_1
    - For non-integer n: Uses connection formula with J_n and J_{-n}
    - For large |z|: Uses asymptotic expansion
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Recurrence Relation
    -------------------
    Bessel functions satisfy the recurrence:

    .. math::

       Y_{n-1}(z) + Y_{n+1}(z) = \frac{2n}{z} Y_n(z)

    This can be used to compute Y_n for any integer order from Y_0 and Y_1.
    Unlike J_n, forward recurrence for Y_n is numerically stable.

    Derivative
    ----------
    The derivative with respect to z satisfies:

    .. math::

       \frac{d}{dz} Y_n(z) = \frac{Y_{n-1}(z) - Y_{n+1}(z)}{2}

    Applications
    ------------
    The Bessel function Y_n appears in many contexts:
    - Electromagnetic wave propagation in cylindrical waveguides
    - Heat conduction problems with cylindrical geometry
    - Vibrations of circular membranes (with mixed boundary conditions)
    - Quantum mechanics (scattering problems)
    - Acoustics and antenna theory

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

       \frac{\partial}{\partial z} Y_n(z) = \frac{Y_{n-1}(z) - Y_{n+1}(z)}{2}

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
        Should be positive real or complex (not on negative real axis).
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Bessel function Y_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel_y(n, z)
    tensor([ 0.0883,  0.1070, -0.1604])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(bessel_y(n0, z), bessel_y_0(z))
    True

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel_y(n, z)
    tensor([-0.4311, -0.1912,  0.0298])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> bessel_y(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = bessel_y(n, z)
    >>> y.backward()
    >>> z.grad  # equals (Y_0(2) - Y_2(2)) / 2
    tensor([0.4813])

    Symmetry for integer n:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> y_pos = bessel_y(n, z)
    >>> y_neg = bessel_y(-n, z)
    >>> torch.allclose(y_neg, y_pos)  # Y_{-2}(z) = Y_2(z)
    True

    Connection formula for non-integer n:

    >>> n = torch.tensor([0.5])
    >>> z = torch.tensor([2.0])
    >>> import math
    >>> j_n = bessel_j(n, z)
    >>> j_neg_n = bessel_j(-n, z)
    >>> expected = (j_n * math.cos(n.item() * math.pi) - j_neg_n) / math.sin(n.item() * math.pi)
    >>> torch.allclose(bessel_y(n, z), expected)
    True

    Singularity at z=0:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([0.0])
    >>> bessel_y(n, z)
    tensor([-inf])

    .. warning:: Singularity at the origin

       Y_n(z) has a logarithmic singularity at z=0 for all orders n.
       The function returns -inf at this point.

    .. warning:: Branch cut for complex arguments

       For complex z with negative real part, Y_n(z) has a branch cut
       along the negative real axis. Results depend on the branch chosen.

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision due to
       the oscillatory nature of Bessel functions and potential numerical
       cancellation. Forward recurrence for Y_n is stable (unlike J_n).

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `bessel_y_0` and
      `bessel_y_1` may provide slightly better accuracy.
    - The implementation uses forward recurrence for integer orders,
      which is numerically stable for Y_n.
    - For non-integer orders, the connection formula with J_n is used.

    See Also
    --------
    bessel_y_0 : Bessel function of the second kind of order zero
    bessel_y_1 : Bessel function of the second kind of order one
    bessel_j : Bessel function of the first kind of general order
    bessel_j_0 : Bessel function of the first kind of order zero
    bessel_j_1 : Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.bessel_y(n, z)
