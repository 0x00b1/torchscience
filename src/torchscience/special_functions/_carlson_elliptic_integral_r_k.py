import torch
from torch import Tensor


def carlson_elliptic_integral_r_k(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_K.

    Computes Carlson's complete symmetric elliptic integral of the first kind.

    Mathematical Definition
    -----------------------
    Carlson's R_K is defined as the integral:

    .. math::

       R_K(x, y) = \frac{1}{2} \int_0^\infty
       \frac{dt}{\sqrt{t(t+x)(t+y)}}

    This is equivalent to R_F with the first argument set to zero:

    .. math::

       R_K(x, y) = R_F(0, x, y)

    Domain
    ------
    - x: non-negative real number (or complex)
    - y: non-negative real number (or complex)
    - At least one of x, y must be non-zero

    Special Values
    --------------
    - R_K(1, 1) = pi/2
    - R_K(0, 1) = infinity (logarithmic singularity)
    - R_K(1, 0) = infinity (logarithmic singularity)
    - R_K(a, a) = pi/(2*sqrt(a)) for a > 0

    Relation to Complete Elliptic Integrals
    ---------------------------------------
    R_K relates to the Legendre complete elliptic integral of the first kind K(k):

    .. math::

       K(k) = R_K(1 - k^2, 1) = R_F(0, 1 - k^2, 1)

    where k is the elliptic modulus.

    Algorithm
    ---------
    Delegates to R_F(0, x, y) which uses the duplication theorem for Carlson
    integrals. The iteration converges quadratically.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_K appears in many contexts:
    - Complete elliptic integral of the first kind
    - Arc length of ellipse
    - Period of simple pendulum
    - Geodesics on ellipsoid
    - Conformal mapping

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are derived from the R_F gradients:

    .. math::

       \frac{\partial R_K}{\partial x} = -\frac{R_D(0, y, x)}{6}

    .. math::

       \frac{\partial R_K}{\partial y} = -\frac{R_D(0, x, y)}{6}

    where R_D is Carlson's elliptic integral R_D.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        First argument. Must be non-negative (or complex).
        Broadcasting with y is supported.
    y : Tensor
        Second argument. Must be non-negative (or complex).
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        Carlson's elliptic integral R_K(x, y) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([1.0])
    >>> carlson_elliptic_integral_r_k(x, y)  # pi/2
    tensor([1.5708])

    R_K(a, a) = pi/(2*sqrt(a)):

    >>> a = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_k(a, a)  # pi/4
    tensor([0.7854])

    Complete elliptic integral K(k) via R_K:

    >>> k = torch.tensor([0.5])  # modulus
    >>> m = 1 - k**2  # parameter
    >>> carlson_elliptic_integral_r_k(m, torch.tensor([1.0]))  # K(0.5)
    tensor([1.6858])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> result = carlson_elliptic_integral_r_k(x, y)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([-0.3682])

    .. warning:: Singularity at x = 0 or y = 0

       When either x = 0 or y = 0, the integral has a logarithmic singularity
       and the function returns inf.

    Notes
    -----
    - R_K is the simplest complete Carlson integral, corresponding to R_F
      with one argument set to zero.
    - The name "R_K" comes from its relation to the complete elliptic integral K.
    - R_K satisfies the homogeneity relation: R_K(ax, ay) = R_K(x, y) / sqrt(a)

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_k(x, y)
