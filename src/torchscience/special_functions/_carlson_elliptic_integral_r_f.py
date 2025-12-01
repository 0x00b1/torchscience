import torch
from torch import Tensor


def carlson_elliptic_integral_r_f(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_F.

    Computes Carlson's symmetric elliptic integral of the first kind.

    Mathematical Definition
    -----------------------
    Carlson's R_F is defined as the symmetric integral:

    .. math::

       R_F(x, y, z) = \frac{1}{2} \int_0^\infty
       \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}}

    This integral converges for all non-negative x, y, z with at most one
    being zero.

    Symmetry Properties
    -------------------
    R_F is completely symmetric in its three arguments:

    .. math::

       R_F(x, y, z) = R_F(y, x, z) = R_F(z, y, x) = \ldots

    Domain
    ------
    - x, y, z: non-negative real numbers (or complex with appropriate branch cuts)
    - At most one of x, y, z may be zero
    - For complex arguments, the principal branch is used

    Special Values
    --------------
    - R_F(0, 1, 1) = pi/2 (complete elliptic integral K(0))
    - R_F(0, 1, 2) = K(1/2) (complete elliptic integral at m=1/2)
    - R_F(x, x, x) = 1/sqrt(x)
    - R_F(0, y, y) = pi/(2*sqrt(y))

    Relation to Legendre Elliptic Integrals
    ---------------------------------------
    The incomplete elliptic integral of the first kind F(phi, k) can be
    expressed using R_F:

    .. math::

       F(\phi, k) = \sin\phi \cdot R_F(\cos^2\phi, 1 - k^2\sin^2\phi, 1)

    The complete elliptic integral of the first kind:

    .. math::

       K(k) = R_F(0, 1 - k^2, 1)

    Algorithm
    ---------
    Uses the duplication theorem for Carlson integrals, which provides
    quadratic convergence. The iteration:

    .. math::

       \lambda_n = \sqrt{x_n}\sqrt{y_n} + \sqrt{y_n}\sqrt{z_n} + \sqrt{z_n}\sqrt{x_n}

    .. math::

       x_{n+1} = (x_n + \lambda_n)/4, \quad y_{n+1} = (y_n + \lambda_n)/4, \quad z_{n+1} = (z_n + \lambda_n)/4

    converges to a common value, and the result is computed from a
    truncated Taylor series.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_F appears in many mathematical and physical contexts:
    - Computing Legendre elliptic integrals with improved numerical stability
    - Geodesic calculations on ellipsoids
    - Pendulum period calculations
    - Conformal mapping
    - Electrostatics and magnetostatics
    - Gravitational potential of ellipsoids

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are computed using the recurrence relation:

    .. math::

       \frac{\partial R_F}{\partial x} = -\frac{1}{6} R_D(y, z, x)

    where R_D is Carlson's elliptic integral of the second kind, with
    similar expressions for the other partial derivatives by symmetry.

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
        Carlson's elliptic integral R_F(x, y, z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0])
    >>> y = torch.tensor([1.0])
    >>> z = torch.tensor([1.0])
    >>> carlson_elliptic_integral_r_f(x, y, z)  # pi/2
    tensor([1.5708])

    Computing complete elliptic integral K(k) = R_F(0, 1-k^2, 1):

    >>> k = torch.tensor([0.5])
    >>> carlson_elliptic_integral_r_f(
    ...     torch.tensor([0.0]),
    ...     1 - k**2,
    ...     torch.tensor([1.0])
    ... )  # K(0.5) approx 1.6858
    tensor([1.6858])

    Symmetric property R_F(a, a, a) = 1/sqrt(a):

    >>> a = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_f(a, a, a)  # 1/sqrt(4) = 0.5
    tensor([0.5000])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> result = carlson_elliptic_integral_r_f(x, y, z)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([-0.0453])

    .. warning:: At most one argument may be zero

       When more than one of x, y, z is zero, the integral diverges
       and the function returns inf.

    .. warning:: Negative real arguments

       For negative real arguments, the function may return NaN or complex
       infinity depending on the branch cut. Use complex inputs for analytic
       continuation to negative values.

    Notes
    -----
    - The Carlson symmetric forms are numerically more stable than the
      traditional Legendre forms for elliptic integrals.
    - R_F satisfies the homogeneity relation: R_F(ax, ay, az) = R_F(x, y, z) / sqrt(a)
    - For complex inputs, the principal branch is used with branch cuts
      along the negative real axis from each argument.

    See Also
    --------
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C (degenerate case)
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_f(x, y, z)
