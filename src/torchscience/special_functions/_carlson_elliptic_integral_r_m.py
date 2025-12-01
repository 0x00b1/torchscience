import torch
from torch import Tensor


def carlson_elliptic_integral_r_m(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_M.

    Computes Carlson's elliptic integral R_M, which is related to the elliptic
    integral of the third kind.

    Mathematical Definition
    -----------------------
    Carlson's R_M is defined as:

    .. math::

       R_M(x, y, z) = \frac{1}{6}\left[(x + y - 2z) R_F(x, y, z)
       + z R_J(x, y, z, z) + 3\sqrt{\frac{xyz}{x + y - z}}\right]

    where R_F is Carlson's elliptic integral of the first kind and R_J is
    Carlson's elliptic integral of the third kind.

    Special Cases
    -------------
    When z = 0:

    .. math::

       R_M(x, y, 0) = \frac{\pi}{4} \frac{x + y}{\sqrt{xy}}

    Domain
    ------
    - x, y, z: non-negative real numbers
    - For complex arguments, appropriate branch cuts apply
    - The denominator (x + y - z) should not be zero unless z = 0

    Special Values
    --------------
    - R_M(0, 0, 0) = 0
    - R_M(x, y, 0) = (pi/4) * (x + y) / sqrt(xy)

    Relation to Elliptic Integrals
    ------------------------------
    R_M appears in expressions for complete and incomplete elliptic integrals
    of the third kind. It provides a unified representation that is
    computationally efficient and numerically stable.

    Algorithm
    ---------
    The implementation uses the relationship to R_F and R_J, which themselves
    use the duplication theorem for Carlson integrals with quadratic convergence.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_M appears in many mathematical and physical contexts:
    - Computing incomplete elliptic integrals of the third kind
    - Geodesic calculations on ellipsoids
    - Electromagnetic field calculations
    - Gravitational potential theory

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
        Carlson's elliptic integral R_M(x, y, z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> carlson_elliptic_integral_r_m(x, y, z)
    tensor([...])

    Special case when z = 0:
    R_M(x, y, 0) = (pi/4) * (x + y) / sqrt(xy)

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([4.0])
    >>> z = torch.tensor([0.0])
    >>> result = carlson_elliptic_integral_r_m(x, y, z)
    >>> expected = (torch.pi / 4) * (1.0 + 4.0) / torch.sqrt(torch.tensor(4.0))
    >>> torch.allclose(result, expected.unsqueeze(0))
    True

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> result = carlson_elliptic_integral_r_m(x, y, z)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([...])

    Notes
    -----
    - R_M is computed via R_F and R_J, inheriting their numerical properties.
    - The function handles the z = 0 special case separately for numerical stability.
    - R_M satisfies homogeneity: R_M(ax, ay, az) = sqrt(a) * R_M(x, y, z)

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_j : Carlson's elliptic integral R_J
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C
    carlson_elliptic_integral_r_g : Carlson's elliptic integral R_G
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_m(x, y, z)
