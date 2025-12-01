import torch
from torch import Tensor


def carlson_elliptic_integral_r_j(
    x: Tensor, y: Tensor, z: Tensor, p: Tensor
) -> Tensor:
    r"""
    Carlson's elliptic integral R_J.

    Computes Carlson's symmetric elliptic integral of the third kind.

    Mathematical Definition
    -----------------------
    Carlson's R_J is defined as the integral:

    .. math::

       R_J(x, y, z, p) = \frac{3}{2} \int_0^\infty
       \frac{dt}{(t+p)\sqrt{(t+x)(t+y)(t+z)}}

    This integral converges for all non-negative x, y, z with at most one
    being zero, p > 0 (or p < 0 with restrictions), and the product xyz > 0.

    Symmetry Properties
    -------------------
    R_J is symmetric in its first three arguments:

    .. math::

       R_J(x, y, z, p) = R_J(y, x, z, p) = R_J(z, y, x, p) = \ldots

    but NOT in the fourth argument p.

    Domain
    ------
    - x, y, z: non-negative real numbers, at most one may be zero
    - p: must be positive (p > 0) for the standard case
    - For complex arguments, appropriate branch cuts apply

    Special Values
    --------------
    - R_J(0, y, y, y) = 3*pi/(2*y^(3/2)*(y+p))
    - R_J(x, x, x, x) = 1/x^(3/2)
    - R_J(x, x, x, p) = R_D(p, p, x) when x = y = z

    Relation to R_D
    ---------------
    R_D is the degenerate case of R_J when the fourth argument equals z:

    .. math::

       R_D(x, y, z) = R_J(x, y, z, z)

    Relation to Legendre Elliptic Integrals
    ---------------------------------------
    The incomplete elliptic integral of the third kind Pi(n, phi, k) can be
    expressed using R_F, R_J:

    .. math::

       \Pi(n, \phi, k) = \sin\phi \cdot R_F(\cos^2\phi, 1 - k^2\sin^2\phi, 1)
                       + \frac{n\sin^3\phi}{3} \cdot R_J(\cos^2\phi, 1 - k^2\sin^2\phi, 1, 1 - n\sin^2\phi)

    The complete elliptic integral of the third kind:

    .. math::

       \Pi(n, k) = R_F(0, 1-k^2, 1) + \frac{n}{3} R_J(0, 1-k^2, 1, 1-n)

    Algorithm
    ---------
    Uses the duplication theorem for Carlson integrals, which provides
    quadratic convergence. The algorithm accumulates R_C correction terms
    during iteration.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_J appears in many mathematical and physical contexts:
    - Computing Legendre elliptic integrals of the third kind
    - Gravitational potential of ellipsoidal bodies
    - Electromagnetic field calculations
    - Geodesics on ellipsoids
    - Pendulum period calculations with large amplitudes

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients involve R_J and R_D evaluated at permuted arguments.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        First argument. Must be non-negative (or complex).
        Broadcasting with y, z, and p is supported.
    y : Tensor
        Second argument. Must be non-negative (or complex).
        Broadcasting with x, z, and p is supported.
    z : Tensor
        Third argument. Must be non-negative (or complex).
        Broadcasting with x, y, and p is supported.
    p : Tensor
        Fourth argument (the "parameter"). Must be positive (or complex).
        Broadcasting with x, y, and z is supported.

    Returns
    -------
    Tensor
        Carlson's elliptic integral R_J(x, y, z, p) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> p = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_j(x, y, z, p)
    tensor([0.1429])

    Relation to R_D when p = z (R_J(x, y, z, z) = R_D(x, y, z)):

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> rj = carlson_elliptic_integral_r_j(x, y, z, z)
    >>> rd = torch.ops.torchscience.carlson_elliptic_integral_r_d(x, y, z)
    >>> torch.allclose(rj, rd, rtol=1e-5)
    True

    Symmetric property R_J(a, a, a, a) = 1/a^(3/2):

    >>> a = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_j(a, a, a, a)  # 1/4^(3/2) = 0.125
    tensor([0.1250])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> p = torch.tensor([4.0])
    >>> result = carlson_elliptic_integral_r_j(x, y, z, p)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([-0.0285])

    .. warning:: p must be positive

       The fourth argument p must be strictly positive for the standard case.
       When p <= 0, the integral requires careful handling of the Cauchy
       principal value.

    .. warning:: Asymmetry in p

       R_J is symmetric in x, y, and z, but NOT in p. Be careful about
       argument ordering when converting from other elliptic integral
       conventions.

    Notes
    -----
    - The Carlson symmetric forms are numerically more stable than the
      traditional Legendre forms for elliptic integrals.
    - R_J satisfies the homogeneity relation: R_J(ax, ay, az, ap) = R_J(x, y, z, p) / a^(3/2)
    - R_J reduces to R_D when p = z: R_J(x, y, z, z) = R_D(x, y, z)

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D (degenerate case of R_J)
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_j(x, y, z, p)
