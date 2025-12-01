import torch
from torch import Tensor


def spherical_bessel_y(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Spherical Bessel function of the second kind of general order n.

    Computes the spherical Bessel function y_n(z) evaluated at each element
    of the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The spherical Bessel function of the second kind of order n is defined as:

    .. math::

       y_n(z) = \sqrt{\frac{\pi}{2z}} Y_{n+1/2}(z)

    where Y_{n+1/2}(z) is the Bessel function of the second kind of order n+1/2.

    For non-negative integer orders, an explicit formula is:

    .. math::

       y_n(z) = (-z)^n \left(\frac{1}{z}\frac{d}{dz}\right)^n \frac{-\cos(z)}{z}

    Special Values
    --------------
    - y_n(0) = -infinity for all n (singular at origin)
    - y_n(NaN) = NaN

    Special Cases
    -------------
    - y_0(z) = -cos(z)/z
    - y_1(z) = -cos(z)/z^2 - sin(z)/z
    - y_2(z) = (-3/z^2 + 1)*cos(z)/z - 3*sin(z)/z^2

    Symmetry
    --------
    For integer n:

    .. math::

       y_n(-z) = (-1)^{n+1} y_n(z)

    Recurrence Relation
    -------------------
    Spherical Bessel functions satisfy the recurrence:

    .. math::

       y_{n-1}(z) + y_{n+1}(z) = \frac{2n+1}{z} y_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: any real or complex number except 0 (argument)
    - y_n(z) is singular at z=0 for all n

    Algorithm
    ---------
    - For integer n = 0: Uses optimized y_0(z) = -cos(z)/z implementation
    - For integer n = 1: Uses optimized y_1(z) implementation
    - For other non-negative integer n: Uses forward recurrence (stable for y_n)
    - For non-integer n: Uses the relation y_n(z) = sqrt(pi/2z) * Y_{n+1/2}(z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The spherical Bessel function y_n appears in many contexts:
    - Quantum mechanics: irregular solutions to the radial Schrodinger equation
    - Scattering theory: phase shifts, partial wave expansion
    - Electrodynamics: outgoing wave solutions in multipole expansions
    - Acoustics: spherical wave propagation
    - Signal processing: spherical harmonic analysis

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

       \frac{\partial}{\partial z} y_n(z) = \frac{n}{z} y_n(z) - y_{n+1}(z)

    Or equivalently:

    .. math::

       \frac{\partial}{\partial z} y_n(z) = y_{n-1}(z) - \frac{n+1}{z} y_n(z)

    The gradient with respect to n is computed numerically since the
    analytical formula involves complex integrals.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the spherical Bessel function. Can be any real or complex number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the spherical Bessel function.
        Can be any real or complex number (singular at z=0).
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The spherical Bessel function y_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> spherical_bessel_y(n, z)
    tensor([-0.5403, -0.3506, -0.2293])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(spherical_bessel_y(n0, z), spherical_bessel_y_0(z))
    True

    Non-integer orders via relation to Bessel Y:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([2.0, 2.0, 2.0])
    >>> spherical_bessel_y(n, z)
    tensor([-0.2081, -0.2851, -0.3479])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> spherical_bessel_y(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_y(n, z)
    >>> y.backward()
    >>> z.grad  # equals (1/z)*y_1(z) - y_2(z)
    tensor([0.2209])

    Recurrence relation verification:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> y_nm1 = spherical_bessel_y(n - 1, z)
    >>> y_n = spherical_bessel_y(n, z)
    >>> y_np1 = spherical_bessel_y(n + 1, z)
    >>> lhs = y_nm1 + y_np1
    >>> rhs = (2*n + 1) / z * y_n
    >>> torch.allclose(lhs, rhs)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> spherical_bessel_y(n, z)
    tensor([-0.6302-0.3878j])

    .. warning:: Singularity at origin

       y_n(z) is singular at z=0 for all values of n. The function returns
       -infinity at z=0.

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision due to
       the oscillatory nature of spherical Bessel functions.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `spherical_bessel_y_0` and
      `spherical_bessel_y_1` are used internally for better accuracy.
    - The spherical Bessel functions arise naturally when solving the
      Helmholtz equation in spherical coordinates using separation of variables.
    - y_n(z) corresponds to the irregular solution (unbounded as z -> 0).
    - Unlike j_n, forward recurrence is stable for computing y_n.

    See Also
    --------
    spherical_bessel_y_0 : Spherical Bessel function y_0
    spherical_bessel_y_1 : Spherical Bessel function y_1
    spherical_bessel_j : Spherical Bessel function of the first kind j_n
    bessel_y : Bessel function of the second kind Y_n
    """
    return torch.ops.torchscience.spherical_bessel_y(n, z)
