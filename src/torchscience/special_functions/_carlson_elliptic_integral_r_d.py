import torch
from torch import Tensor


def carlson_elliptic_integral_r_d(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_D.

    Computes Carlson's symmetric elliptic integral of the second kind.

    Mathematical Definition
    -----------------------
    Carlson's R_D is defined as the integral:

    .. math::

       R_D(x, y, z) = \frac{3}{2} \int_0^\infty
       \frac{dt}{(t+z)\sqrt{(t+x)(t+y)(t+z)}}

    This integral converges for all non-negative x, y, z with at most one
    of x or y being zero, and z > 0.

    Symmetry Properties
    -------------------
    R_D is symmetric only in its first two arguments:

    .. math::

       R_D(x, y, z) = R_D(y, x, z)

    but NOT in the third argument z.

    Domain
    ------
    - x, y: non-negative real numbers, at most one may be zero
    - z: must be positive (z > 0)
    - For complex arguments, appropriate branch cuts apply

    Special Values
    --------------
    - R_D(0, 1, 1) = 3(pi/4 - 1/2)
    - R_D(0, 2, 1) = 3(K(1/sqrt(2)) - E(1/sqrt(2)))
    - R_D(x, x, x) = 1/x^(3/2)
    - R_D(0, y, y) = 3*pi/(4*y^(3/2))

    Relation to Legendre Elliptic Integrals
    ---------------------------------------
    The incomplete elliptic integral of the second kind E(phi, k) can be
    expressed using R_F and R_D:

    .. math::

       E(\phi, k) = \sin\phi \cdot R_F(\cos^2\phi, 1 - k^2\sin^2\phi, 1)
                  - \frac{k^2\sin^3\phi}{3} \cdot R_D(\cos^2\phi, 1 - k^2\sin^2\phi, 1)

    The complete elliptic integral of the second kind:

    .. math::

       E(k) = R_F(0, 1-k^2, 1) - \frac{k^2}{3} R_D(0, 1-k^2, 1)

    Relation to R_F
    ---------------
    R_D is related to the partial derivative of R_F:

    .. math::

       \frac{\partial R_F(x, y, z)}{\partial z} = -\frac{1}{6} R_D(x, y, z)

    Algorithm
    ---------
    Uses the duplication theorem for Carlson integrals, which provides
    quadratic convergence. Similar to R_F but with an additional sum
    accumulated during the iteration.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_D appears in many mathematical and physical contexts:
    - Computing Legendre elliptic integrals of the second kind
    - Arc length of ellipses
    - Surface area of ellipsoids
    - Gravitational and electrostatic potentials
    - Computing gradients of R_F (used internally for autograd)

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients involve R_D evaluated at permuted arguments and
    higher-order Carlson integrals.

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
        Third argument. Must be positive (or complex with positive real part).
        Broadcasting with x and y is supported.

    Returns
    -------
    Tensor
        Carlson's elliptic integral R_D(x, y, z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0])
    >>> y = torch.tensor([1.0])
    >>> z = torch.tensor([1.0])
    >>> carlson_elliptic_integral_r_d(x, y, z)
    tensor([1.7972])

    Computing complete elliptic integral E(k) using R_F and R_D:

    >>> k = torch.tensor([0.5])
    >>> k2 = k**2
    >>> x = torch.tensor([0.0])
    >>> y = 1 - k2
    >>> z = torch.tensor([1.0])
    >>> rf = torch.ops.torchscience.carlson_elliptic_integral_r_f(x, y, z)
    >>> rd = carlson_elliptic_integral_r_d(x, y, z)
    >>> E_k = rf - (k2/3) * rd  # E(0.5)
    >>> E_k
    tensor([1.4675])

    Symmetric property R_D(a, a, a) = 1/a^(3/2):

    >>> a = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_d(a, a, a)  # 1/4^(3/2) = 0.125
    tensor([0.1250])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> result = carlson_elliptic_integral_r_d(x, y, z)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([-0.0288])

    .. warning:: z must be positive

       Unlike x and y, the third argument z must be strictly positive.
       When z = 0, the integral diverges and the function returns inf.

    .. warning:: Asymmetry in z

       R_D is symmetric in x and y, but NOT in z. Be careful about
       argument ordering when converting from other elliptic integral
       conventions.

    Notes
    -----
    - The Carlson symmetric forms are numerically more stable than the
      traditional Legendre forms for elliptic integrals.
    - R_D satisfies the homogeneity relation: R_D(ax, ay, az) = R_D(x, y, z) / a^(3/2)
    - R_D is related to R_J when one argument equals another: R_D(x, y, z) = R_J(x, y, z, z)

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C (degenerate case)
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_d(x, y, z)
