import torch
from torch import Tensor


def spherical_bessel_k(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Modified spherical Bessel function of the second kind of general order n.

    Computes the modified spherical Bessel function k_n(z) evaluated at each element
    of the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The modified spherical Bessel function of the second kind of order n is defined as:

    .. math::

       k_n(z) = \sqrt{\frac{\pi}{2z}} K_{n+1/2}(z)

    where K_{n+1/2}(z) is the modified Bessel function of the second kind of order n+1/2.

    For non-negative integer orders, an explicit formula is:

    .. math::

       k_n(z) = \frac{\pi}{2} e^{-z} \sum_{k=0}^{n} \frac{(n+k)!}{k!(n-k)!} \frac{1}{(2z)^{k+1}}

    Special Values
    --------------
    - k_n(0) = infinity for all n (pole at origin)
    - k_n(infinity) = 0 (exponential decay)
    - k_n(NaN) = NaN

    Special Cases
    -------------
    - k_0(z) = (pi/2) * e^(-z) / z
    - k_1(z) = (pi/2) * (1 + z) * e^(-z) / z^2

    Recurrence Relation
    -------------------
    Modified spherical Bessel functions of the second kind satisfy the recurrence:

    .. math::

       k_{n-1}(z) + k_{n+1}(z) = \frac{2n+1}{z} k_n(z)

    Derivative Formula
    ------------------
    The derivative with respect to z is:

    .. math::

       \frac{d}{dz} k_n(z) = -k_{n-1}(z) - \frac{n}{z} k_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: positive real numbers (for real implementation)
    - z: complex numbers (for complex implementation)
    - k_n(z) is singular at z=0 (pole)

    Algorithm
    ---------
    - For integer n = 0: Uses optimized k_0(z) = (pi/2) * exp(-z) / z implementation
    - For integer n = 1: Uses optimized k_1(z) implementation
    - For other non-negative integer n: Uses forward recurrence (stable for k_n)
    - For non-integer n: Uses the relation k_n(z) = sqrt(pi/2z) * K_{n+1/2}(z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified spherical Bessel function k_n appears in many contexts:
    - Quantum mechanics: exponentially decaying wave functions
    - Heat conduction: spherically symmetric problems with decay
    - Diffusion: spherical diffusion with absorption
    - Electrodynamics: evanescent fields in spherical geometries
    - Signal processing: filter design with exponential decay

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

       \frac{\partial}{\partial z} k_n(z) = -k_{n-1}(z) - \frac{n}{z} k_n(z)

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
        Must be positive for real inputs.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The modified spherical Bessel function k_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> spherical_bessel_k(n, z)
    tensor([0.5779, 0.1520, 0.0369])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(spherical_bessel_k(n0, z), spherical_bessel_k_0(z))
    True

    Value at origin (singular):

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([0.0, 0.0, 0.0])
    >>> spherical_bessel_k(n, z)
    tensor([inf, inf, inf])

    Non-integer orders via relation to modified Bessel K:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([2.0, 2.0, 2.0])
    >>> spherical_bessel_k(n, z)
    tensor([0.0958, 0.0500, 0.0315])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> spherical_bessel_k(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_k(n, z)
    >>> y.backward()
    >>> z.grad  # equals -k_0(z) - (1/z)*k_1(z)
    tensor([-0.1712])

    Recurrence relation verification:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> k_nm1 = spherical_bessel_k(n - 1, z)
    >>> k_n = spherical_bessel_k(n, z)
    >>> k_np1 = spherical_bessel_k(n + 1, z)
    >>> lhs = k_nm1 + k_np1
    >>> rhs = (2*n + 1) / z * k_n
    >>> torch.allclose(lhs, rhs)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> spherical_bessel_k(n, z)
    tensor([0.4284-0.3812j])

    .. warning:: Numerical precision

       k_n(z) decays exponentially for large z, which may lead to underflow
       for very large arguments. For small z, k_n(z) grows rapidly (pole at origin),
       which may lead to overflow.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `spherical_bessel_k_0` and
      `spherical_bessel_k_1` are used internally for better accuracy.
    - The modified spherical Bessel functions arise naturally when solving the
      modified Helmholtz equation in spherical coordinates using separation of variables.
    - k_n(z) corresponds to the exponentially decaying solution as z -> infinity.
    - k_n(z) is always positive for real positive z and non-negative integer n.

    See Also
    --------
    spherical_bessel_k_0 : Modified spherical Bessel function k_0
    spherical_bessel_k_1 : Modified spherical Bessel function k_1
    modified_bessel_k : Modified Bessel function of the second kind K_n
    spherical_bessel_i : Modified spherical Bessel function of the first kind i_n
    """
    return torch.ops.torchscience.spherical_bessel_k(n, z)
