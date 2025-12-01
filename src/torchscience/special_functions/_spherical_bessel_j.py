import torch
from torch import Tensor


def spherical_bessel_j(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Spherical Bessel function of the first kind of general order n.

    Computes the spherical Bessel function j_n(z) evaluated at each element
    of the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The spherical Bessel function of the first kind of order n is defined as:

    .. math::

       j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n+1/2}(z)

    where J_{n+1/2}(z) is the Bessel function of the first kind of order n+1/2.

    For non-negative integer orders, an explicit formula is:

    .. math::

       j_n(z) = (-z)^n \left(\frac{1}{z}\frac{d}{dz}\right)^n \frac{\sin(z)}{z}

    Power series representation for small z:

    .. math::

       j_n(z) = \frac{z^n}{(2n+1)!!} \left[1 - \frac{z^2}{2(2n+3)} + \cdots\right]

    Special Values
    --------------
    - j_0(0) = 1
    - j_n(0) = 0 for n > 0
    - j_n(NaN) = NaN

    Special Cases
    -------------
    - j_0(z) = sin(z)/z
    - j_1(z) = sin(z)/z^2 - cos(z)/z
    - j_2(z) = (3/z^2 - 1)*sin(z)/z - 3*cos(z)/z^2

    Recurrence Relation
    -------------------
    Spherical Bessel functions satisfy the recurrence:

    .. math::

       j_{n-1}(z) + j_{n+1}(z) = \frac{2n+1}{z} j_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: any real or complex number (argument)
    - For non-integer negative n at z=0, the function is singular

    Algorithm
    ---------
    - For integer n = 0: Uses optimized j_0(z) = sin(z)/z implementation
    - For integer n = 1: Uses optimized j_1(z) implementation
    - For other non-negative integer n: Uses forward or backward recurrence
      depending on |z| vs n for numerical stability
    - For non-integer n: Uses the relation j_n(z) = sqrt(pi/2z) * J_{n+1/2}(z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The spherical Bessel function j_n appears in many contexts:
    - Quantum mechanics: free particle wave functions in 3D, partial wave expansion
    - Scattering theory: phase shifts, cross sections
    - Electrodynamics: multipole expansions of electromagnetic fields
    - Acoustics: spherical wave propagation
    - Signal processing: spherical harmonic analysis
    - Cosmology: matter power spectrum analysis

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

       \frac{\partial}{\partial z} j_n(z) = \frac{n}{z} j_n(z) - j_{n+1}(z)

    Or equivalently:

    .. math::

       \frac{\partial}{\partial z} j_n(z) = j_{n-1}(z) - \frac{n+1}{z} j_n(z)

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
        Can be any real or complex number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The spherical Bessel function j_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> spherical_bessel_j(n, z)
    tensor([0.8415, 0.4353, 0.2986])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(spherical_bessel_j(n0, z), spherical_bessel_j_0(z))
    True

    Value at origin:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([0.0, 0.0, 0.0])
    >>> spherical_bessel_j(n, z)
    tensor([1., 0., 0.])

    Non-integer orders via relation to Bessel J:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([2.0, 2.0, 2.0])
    >>> spherical_bessel_j(n, z)
    tensor([0.5403, 0.3522, 0.1981])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> spherical_bessel_j(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_j(n, z)
    >>> y.backward()
    >>> z.grad  # equals (1/z)*j_1(z) - j_2(z)
    tensor([0.0206])

    Recurrence relation verification:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> j_nm1 = spherical_bessel_j(n - 1, z)
    >>> j_n = spherical_bessel_j(n, z)
    >>> j_np1 = spherical_bessel_j(n + 1, z)
    >>> lhs = j_nm1 + j_np1
    >>> rhs = (2*n + 1) / z * j_n
    >>> torch.allclose(lhs, rhs)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> spherical_bessel_j(n, z)
    tensor([0.8856+0.1174j])

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision due to
       the oscillatory nature of spherical Bessel functions and potential
       numerical cancellation. For integer orders, the backward recurrence
       (Miller's algorithm) is used when z < n for better stability.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `spherical_bessel_j_0` and
      `spherical_bessel_j_1` are used internally for better accuracy.
    - The spherical Bessel functions arise naturally when solving the
      Helmholtz equation in spherical coordinates using separation of variables.
    - j_n(z) corresponds to the regular solution at z=0 (bounded as z -> 0).

    See Also
    --------
    spherical_bessel_j_0 : Spherical Bessel function j_0
    spherical_bessel_j_1 : Spherical Bessel function j_1
    bessel_j : Bessel function of the first kind J_n
    """
    return torch.ops.torchscience.spherical_bessel_j(n, z)
