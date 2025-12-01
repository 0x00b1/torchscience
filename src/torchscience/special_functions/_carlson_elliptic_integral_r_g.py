import torch
from torch import Tensor


def carlson_elliptic_integral_r_g(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_G.

    Computes Carlson's symmetric elliptic integral R_G, which represents the
    average of sqrt over an ellipsoid.

    Mathematical Definition
    -----------------------
    Carlson's R_G is defined as the surface integral:

    .. math::

       R_G(x, y, z) = \frac{1}{4\pi} \iint
       \sqrt{x\cos^2\theta\sin^2\phi + y\sin^2\theta\sin^2\phi + z\cos^2\phi}
       \sin\phi \, d\theta \, d\phi

    where the integration is over the surface of a unit sphere.

    It can also be expressed in terms of R_F and R_D:

    .. math::

       R_G(x, y, z) = \frac{1}{2}\left[z R_F(x, y, z)
       - \frac{(x-z)(y-z)}{3} R_D(x, y, z) + \sqrt{\frac{xy}{z}}\right]

    Symmetry Properties
    -------------------
    R_G is fully symmetric in all three arguments:

    .. math::

       R_G(x, y, z) = R_G(y, x, z) = R_G(z, y, x) = \ldots

    Domain
    ------
    - x, y, z: non-negative real numbers
    - At most one of the arguments may be zero
    - For complex arguments, appropriate branch cuts apply

    Special Values
    --------------
    - R_G(0, 0, 0) = 0
    - R_G(x, x, x) = sqrt(x)
    - R_G(0, y, y) = (pi/4) * sqrt(y)
    - R_G(0, 0, z) = (1/2) * sqrt(z)

    Relation to Complete Elliptic Integrals
    ---------------------------------------
    The complete elliptic integral of the second kind E(k) is related to R_G:

    .. math::

       E(k) = 2 R_G(0, 1-k^2, 1)

    The complete elliptic integral of the first kind K(k):

    .. math::

       K(k) = R_F(0, 1-k^2, 1)

    Physical Interpretation
    -----------------------
    R_G(x, y, z) represents the average of sqrt(x*cos^2*sin^2 + y*sin^2*sin^2 + z*cos^2)
    over the surface of a unit sphere. This has applications in:

    - Computing surface area of ellipsoids
    - Electromagnetic field calculations
    - Gravitational potential theory

    Algorithm
    ---------
    The implementation uses the relationship to R_F and R_D, which themselves
    use the duplication theorem for Carlson integrals with quadratic convergence.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_G appears in many mathematical and physical contexts:
    - Computing complete and incomplete elliptic integrals
    - Surface area of ellipsoids
    - Arc length of ellipses
    - Electromagnetic field calculations
    - Gravitational potential of ellipsoidal bodies

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        First argument. Must be non-negative (or complex).
        Broadcasting with y and z is supported.
    y : Tensor
        Second argument. Must be non-negative (or complex).
        Broadcasting with x and z is supported.
    z : Tensor
        Third argument. Must be non-negative (or complex).
        Broadcasting with x and y is supported.

    Returns
    -------
    Tensor
        Carlson's elliptic integral R_G(x, y, z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> carlson_elliptic_integral_r_g(x, y, z)
    tensor([1.4011])

    Symmetric property R_G(a, a, a) = sqrt(a):

    >>> a = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_g(a, a, a)  # sqrt(4) = 2.0
    tensor([2.0000])

    Relation to complete elliptic integral E(k):
    E(k) = 2 * R_G(0, 1-k^2, 1)

    >>> k = torch.tensor([0.5])
    >>> rg = carlson_elliptic_integral_r_g(
    ...     torch.tensor([0.0]),
    ...     1 - k**2,
    ...     torch.tensor([1.0])
    ... )
    >>> E_k = 2 * rg  # Complete elliptic integral of the second kind
    >>> E_k
    tensor([1.4675])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> result = carlson_elliptic_integral_r_g(x, y, z)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([0.1945])

    Notes
    -----
    - R_G is the only symmetric Carlson integral that is bounded for all
      non-negative arguments (including the case when one or more is zero).
    - R_G satisfies the homogeneity relation: R_G(ax, ay, az) = sqrt(a) * R_G(x, y, z)
    - Unlike R_F and R_D, R_G is finite even when one argument is zero.

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C
    carlson_elliptic_integral_r_j : Carlson's elliptic integral R_J
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_g(x, y, z)
