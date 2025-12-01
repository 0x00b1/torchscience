import torch
from torch import Tensor


def carlson_elliptic_integral_r_c(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_C.

    Computes Carlson's degenerate symmetric elliptic integral.

    Mathematical Definition
    -----------------------
    Carlson's R_C is defined as the integral:

    .. math::

       R_C(x, y) = \frac{1}{2} \int_0^\infty
       \frac{dt}{(t+y)\sqrt{t+x}}

    This is a degenerate case of R_F where two arguments are equal:

    .. math::

       R_C(x, y) = R_F(x, y, y)

    Domain
    ------
    - x: non-negative real number (or complex)
    - y: non-zero real number (or complex with non-zero value)
    - For y < 0, the Cauchy principal value is computed

    Special Values
    --------------
    - R_C(0, 1) = pi/2
    - R_C(1, 1) = 1
    - R_C(0, y) = pi/(2*sqrt(y)) for y > 0
    - R_C(x, x) = 1/sqrt(x)
    - R_C(x, 0) = infinity (pole at y = 0)

    Relation to Elementary Functions
    --------------------------------
    R_C can be expressed in terms of elementary functions:

    For y > 0:

    .. math::

       R_C(x, y) = \begin{cases}
       \frac{\arccos\sqrt{x/y}}{\sqrt{y-x}} & \text{if } x < y \\
       \frac{1}{\sqrt{y}} & \text{if } x = y \\
       \frac{\text{arccosh}\sqrt{x/y}}{\sqrt{x-y}} & \text{if } x > y
       \end{cases}

    For y < 0 (Cauchy principal value):

    .. math::

       R_C(x, y) = \sqrt{\frac{x}{x-y}} R_C(x-y, -y)

    Algorithm
    ---------
    Uses the duplication theorem for Carlson integrals. The iteration
    converges quadratically and is simpler than the general R_F case.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_C appears in many contexts:
    - Building block for other Carlson integrals (R_J, R_G)
    - Inverse trigonometric and hyperbolic functions (generalized)
    - Logarithms of complex numbers (generalized)
    - Electrostatic potential of a charged disk
    - Computing arc lengths

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are:

    .. math::

       \frac{\partial R_C}{\partial x} = -\frac{1}{2x} \left[ R_C(x, y) - \frac{1}{\sqrt{x}} \right] / (x - y)

    .. math::

       \frac{\partial R_C}{\partial y} = \frac{1}{2(y-x)} \left[ \frac{1}{\sqrt{x}} - R_C(x, y) \right] - \frac{R_C(x, y)}{2y}

    (with appropriate limits when x = y)

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        First argument. Must be non-negative (or complex).
        Broadcasting with y is supported.
    y : Tensor
        Second argument. Must be non-zero.
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        Carlson's elliptic integral R_C(x, y) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0])
    >>> y = torch.tensor([1.0])
    >>> carlson_elliptic_integral_r_c(x, y)  # pi/2
    tensor([1.5708])

    R_C(1, 1) = 1:

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([1.0])
    >>> carlson_elliptic_integral_r_c(x, y)
    tensor([1.0000])

    Relation to arccos: R_C(0, y) = pi/(2*sqrt(y)):

    >>> y = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_c(torch.tensor([0.0]), y)  # pi/4
    tensor([0.7854])

    R_C(x, x) = 1/sqrt(x):

    >>> a = torch.tensor([4.0])
    >>> carlson_elliptic_integral_r_c(a, a)  # 1/sqrt(4) = 0.5
    tensor([0.5000])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> result = carlson_elliptic_integral_r_c(x, y)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([-0.1963])

    .. warning:: y must be non-zero

       When y = 0, the integral diverges and the function returns inf.

    .. warning:: Cauchy principal value for y < 0

       For negative y, the Cauchy principal value is computed. Results
       may differ from naive numerical integration.

    Notes
    -----
    - R_C is the simplest Carlson integral and serves as a building block
      for R_J and R_G.
    - Despite being expressible in elementary functions, the Carlson form
      provides better numerical stability across different argument ranges.
    - R_C satisfies the homogeneity relation: R_C(ax, ay) = R_C(x, y) / sqrt(a)

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_c(x, y)
