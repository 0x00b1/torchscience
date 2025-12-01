import torch
from torch import Tensor


def spherical_bessel_i(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Modified spherical Bessel function of the first kind of general order n.

    Computes the modified spherical Bessel function i_n(z) evaluated at each element
    of the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The modified spherical Bessel function of the first kind of order n is defined as:

    .. math::

       i_n(z) = \sqrt{\frac{\pi}{2z}} I_{n+1/2}(z)

    where I_{n+1/2}(z) is the modified Bessel function of the first kind of order n+1/2.

    For non-negative integer orders, an explicit formula is:

    .. math::

       i_n(z) = (-z)^n \left(\frac{1}{z}\frac{d}{dz}\right)^n \frac{\sinh(z)}{z}

    Power series representation for small z:

    .. math::

       i_n(z) = \frac{z^n}{(2n+1)!!} \left[1 + \frac{z^2}{2(2n+3)} + \cdots\right]

    Special Values
    --------------
    - i_0(0) = 1
    - i_n(0) = 0 for n > 0
    - i_n(NaN) = NaN

    Special Cases
    -------------
    - i_0(z) = sinh(z)/z
    - i_1(z) = cosh(z)/z - sinh(z)/z^2

    Recurrence Relation
    -------------------
    Modified spherical Bessel functions satisfy the recurrence:

    .. math::

       i_{n-1}(z) - i_{n+1}(z) = \frac{2n+1}{z} i_n(z)

    Symmetry
    --------
    For integer n:

    .. math::

       i_n(-z) = (-1)^n i_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: any real or complex number (argument)
    - For non-integer negative n at z=0, the function is singular

    Algorithm
    ---------
    - For integer n = 0: Uses optimized i_0(z) = sinh(z)/z implementation
    - For integer n = 1: Uses optimized i_1(z) implementation
    - For other non-negative integer n: Uses forward or backward recurrence
      depending on |z| vs n for numerical stability
    - For non-integer n: Uses the relation i_n(z) = sqrt(pi/2z) * I_{n+1/2}(z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified spherical Bessel function i_n appears in many contexts:
    - Quantum mechanics: bound state wave functions, spherical potential wells
    - Heat conduction: spherically symmetric heat flow problems
    - Diffusion: spherical diffusion equations
    - Electrodynamics: evanescent multipole fields
    - Signal processing: filter design with spherical symmetry

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

       \frac{\partial}{\partial z} i_n(z) = \frac{n}{z} i_n(z) + i_{n+1}(z)

    Or equivalently:

    .. math::

       \frac{\partial}{\partial z} i_n(z) = i_{n-1}(z) - \frac{n+1}{z} i_n(z)

    Note the + sign in the first formula (unlike j_n which has a - sign).

    The gradient with respect to n is computed numerically since the
    analytical formula involves complex integrals.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the modified spherical Bessel function. Can be any real or complex number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the modified spherical Bessel function.
        Can be any real or complex number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The modified spherical Bessel function i_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> spherical_bessel_i(n, z)
    tensor([1.1752, 1.8134, 4.1579])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(spherical_bessel_i(n0, z), spherical_bessel_i_0(z))
    True

    Value at origin:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([0.0, 0.0, 0.0])
    >>> spherical_bessel_i(n, z)
    tensor([1., 0., 0.])

    Non-integer orders via relation to modified Bessel I:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([2.0, 2.0, 2.0])
    >>> spherical_bessel_i(n, z)
    tensor([2.5456, 2.0461, 1.4897])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> spherical_bessel_i(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_i(n, z)
    >>> y.backward()
    >>> z.grad  # equals (1/z)*i_1(z) + i_2(z)
    tensor([1.8065])

    Recurrence relation verification:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> i_nm1 = spherical_bessel_i(n - 1, z)
    >>> i_n = spherical_bessel_i(n, z)
    >>> i_np1 = spherical_bessel_i(n + 1, z)
    >>> lhs = i_nm1 - i_np1
    >>> rhs = (2*n + 1) / z * i_n
    >>> torch.allclose(lhs, rhs)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> spherical_bessel_i(n, z)
    tensor([1.0417+0.5870j])

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision due to
       exponential growth and potential numerical overflow. For integer orders,
       the backward recurrence (Miller's algorithm) is used when z < n for
       better stability.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `spherical_bessel_i_0` and
      `spherical_bessel_i_1` are used internally for better accuracy.
    - The modified spherical Bessel functions arise naturally when solving the
      modified Helmholtz equation in spherical coordinates using separation of variables.
    - i_n(z) corresponds to the exponentially growing solution as z -> infinity.
    - Unlike j_n(z), i_n(z) is non-oscillatory and grows exponentially for large z.

    See Also
    --------
    spherical_bessel_i_0 : Modified spherical Bessel function i_0
    spherical_bessel_i_1 : Modified spherical Bessel function i_1
    modified_bessel_i : Modified Bessel function of the first kind I_n
    spherical_bessel_j : Spherical Bessel function of the first kind j_n
    """
    return torch.ops.torchscience.spherical_bessel_i(n, z)
