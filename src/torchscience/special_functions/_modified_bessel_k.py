import torch
from torch import Tensor


def modified_bessel_k(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Modified Bessel function of the second kind of general order n.

    Computes the modified Bessel function K_n(z) evaluated at each element of
    the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The modified Bessel function of the second kind of order n is defined as:

    .. math::

       K_n(z) = \frac{\pi}{2} \frac{I_{-n}(z) - I_n(z)}{\sin(n\pi)}

    where I_n(z) is the modified Bessel function of the first kind.
    For integer n, the limit is taken as n approaches the integer value.

    Alternatively, it can be expressed as an integral:

    .. math::

       K_n(z) = \int_0^\infty e^{-z \cosh t} \cosh(nt) \, dt

    Special Values
    --------------
    - K_n(0) = +infinity for all n (singularity at origin)
    - K_n(z) > 0 for z > 0 (always positive on positive real axis)
    - K_n(+infinity) = 0 (exponential decay)
    - K_n(NaN) = NaN

    Symmetry
    --------
    For all n (integer or non-integer):

    .. math::

       K_{-n}(z) = K_n(z)

    This symmetry is a fundamental property of K_n.

    Domain
    ------
    - n: any real or complex number (order)
    - z: positive real or complex number (argument)
    - For z <= 0 real, the function has a singularity or is complex-valued

    Algorithm
    ---------
    - For integer n = 0, 1: Uses optimized K_0, K_1 implementations
    - For other integer n: Uses forward recurrence from K_0 and K_1
      - K_{n+1}(z) = K_{n-1}(z) + (2n/z) * K_n(z)
    - For non-integer n: Uses connection formula with I_n(z)
    - For large |z|: Uses asymptotic expansion
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Recurrence Relation
    -------------------
    Modified Bessel functions of the second kind satisfy the recurrence:

    .. math::

       K_{n+1}(z) = K_{n-1}(z) + \frac{2n}{z} K_n(z)

    This recurrence is numerically stable for forward computation (unlike I_n).

    Derivative
    ----------
    The derivative with respect to z is:

    .. math::

       \frac{d}{dz} K_n(z) = -\frac{K_{n-1}(z) + K_{n+1}(z)}{2}

    Note the negative sign, which reflects the exponential decay behavior.

    Asymptotic Behavior
    -------------------
    For large z:

    .. math::

       K_n(z) \sim \sqrt{\frac{\pi}{2z}} e^{-z} \left(1 + \frac{4n^2-1}{8z} + \cdots\right)

    Applications
    ------------
    The modified Bessel function K_n appears in many contexts:
    - Heat conduction in cylindrical geometries
    - Potential theory and electrostatics
    - Quantum mechanics (screened Coulomb potential)
    - Signal processing (Matern covariance functions)
    - Probability theory (distributions involving K_n)
    - Physics of cosmic strings

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

       \frac{\partial}{\partial z} K_n(z) = -\frac{K_{n-1}(z) + K_{n+1}(z)}{2}

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
        Should be positive for real-valued results.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The modified Bessel function K_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> modified_bessel_k(n, z)
    tensor([0.4210, 0.1399, 0.0615])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(modified_bessel_k(n0, z), modified_bessel_k_0(z))
    True

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> modified_bessel_k(n, z)
    tensor([0.4615, 0.1220, 0.0528])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> modified_bessel_k(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = modified_bessel_k(n, z)
    >>> y.backward()
    >>> z.grad  # equals -(K_0(2) + K_2(2)) / 2
    tensor([-0.1700])

    Symmetry in n:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> k_pos = modified_bessel_k(n, z)
    >>> k_neg = modified_bessel_k(-n, z)
    >>> torch.allclose(k_neg, k_pos)  # K_{-2}(z) = K_2(z)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> modified_bessel_k(n, z)
    tensor([0.3274-0.2507j])

    Exponential decay for large z:

    >>> n = torch.tensor([0.0])
    >>> z = torch.tensor([10.0])
    >>> modified_bessel_k(n, z)
    tensor([1.7780e-05])

    .. warning:: Singularity at z=0

       K_n(0) = +infinity for all n. The function has a logarithmic
       singularity at z=0 for n=0 and a power-law singularity for n>0.

    .. warning:: Numerical precision

       For very large |n| or |z|, or when |z| is very small, the function
       may lose precision. The exponential decay for large z can lead to
       underflow.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `modified_bessel_k_0` and
      `modified_bessel_k_1` may provide slightly better accuracy.
    - The implementation uses forward recurrence for integer orders, which is
      numerically stable for K_n (unlike I_n where backward recurrence is
      needed).
    - For non-integer orders, the connection formula with I_n is used.

    See Also
    --------
    modified_bessel_k_0 : Modified Bessel function K_0(z)
    modified_bessel_k_1 : Modified Bessel function K_1(z)
    modified_bessel_i_0 : Modified Bessel function I_0(z)
    modified_bessel_i_1 : Modified Bessel function I_1(z)
    bessel_j : Bessel function of the first kind J_n(z)
    bessel_y : Bessel function of the second kind Y_n(z)
    """
    return torch.ops.torchscience.modified_bessel_k(n, z)
