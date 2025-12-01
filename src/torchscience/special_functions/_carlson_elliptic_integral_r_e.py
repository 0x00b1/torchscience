import torch
from torch import Tensor


def carlson_elliptic_integral_r_e(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    r"""
    Carlson's elliptic integral R_E.

    Computes Carlson's elliptic integral R_E, which is related to R_D and
    appears in various applications involving elliptic integrals.

    Mathematical Definition
    -----------------------
    Carlson's R_E is defined as:

    .. math::

       R_E(x, y, z) = \frac{3}{2} \int_0^\infty
       \frac{t \, dt}{(t+z)\sqrt{(t+x)(t+y)(t+z)}}

    Key Relationships
    -----------------
    R_E can be expressed in terms of R_D:

    .. math::

       R_E(x, y, z) = \frac{3}{2} z R_D(x, y, z) + \sqrt{\frac{xy}{z}}

    Or equivalently:

    .. math::

       R_E(x, y, z) = R_D(y, z, x) + R_D(z, x, y) + 3\sqrt{\frac{xyz}{xy + xz + yz}}

    Symmetry Properties
    -------------------
    R_E is symmetric in x and y, but z plays a distinguished role:

    .. math::

       R_E(x, y, z) = R_E(y, x, z)

    Note that R_E is NOT fully symmetric - permuting z with x or y gives different values.

    Domain
    ------
    - x, y: non-negative real numbers
    - z: positive real number (z > 0 required to avoid divergence)
    - For complex arguments, appropriate branch cuts apply

    Special Values
    --------------
    - R_E(x, y, z) diverges as z approaches 0
    - R_E(0, 0, z) = (3/2) * z^(-1/2)
    - R_E(a, a, a) = 1

    Homogeneity
    -----------
    R_E satisfies the homogeneity relation:

    .. math::

       R_E(\lambda x, \lambda y, \lambda z) = \lambda^{-1/2} R_E(x, y, z)

    Algorithm
    ---------
    The implementation uses the relationship to R_D:
    R_E(x,y,z) = (3/2)*z*R_D(x,y,z) + sqrt(xy/z)

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Carlson's R_E appears in various mathematical and physical contexts:
    - Evaluation of elliptic integrals
    - Electromagnetic field calculations
    - Computing arc lengths of ellipses
    - Gravitational potential calculations

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
        Third argument. Must be positive (or complex with non-zero modulus).
        Broadcasting with x and y is supported.

    Returns
    -------
    Tensor
        Carlson's elliptic integral R_E(x, y, z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> carlson_elliptic_integral_r_e(x, y, z)
    tensor([0.6420])

    Relationship to R_D: R_E(x,y,z) = (3/2)*z*R_D(x,y,z) + sqrt(xy/z)

    >>> from torchscience.special_functions import carlson_elliptic_integral_r_d
    >>> x = torch.tensor([1.0])
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> rd = carlson_elliptic_integral_r_d(x, y, z)
    >>> re_from_rd = 1.5 * z * rd + torch.sqrt(x * y / z)
    >>> re_direct = carlson_elliptic_integral_r_e(x, y, z)
    >>> torch.allclose(re_from_rd, re_direct)
    True

    Homogeneity: R_E(kx, ky, kz) = k^(-1/2) * R_E(x, y, z)

    >>> k = torch.tensor([4.0])
    >>> re1 = carlson_elliptic_integral_r_e(k * x, k * y, k * z)
    >>> re2 = carlson_elliptic_integral_r_e(x, y, z) / torch.sqrt(k)
    >>> torch.allclose(re1, re2)
    True

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> result = carlson_elliptic_integral_r_e(x, y, z)
    >>> result.backward()
    >>> x.grad  # Gradient w.r.t. x
    tensor([...])

    Notes
    -----
    - R_E is related to but distinct from the complete elliptic integral E(k).
    - Unlike R_G which is finite at zero arguments, R_E requires z > 0.
    - R_E is symmetric only in its first two arguments (x and y).

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_d : Carlson's elliptic integral R_D
    carlson_elliptic_integral_r_c : Carlson's elliptic integral R_C
    carlson_elliptic_integral_r_g : Carlson's elliptic integral R_G
    carlson_elliptic_integral_r_j : Carlson's elliptic integral R_J
    """
    return torch.ops.torchscience.carlson_elliptic_integral_r_e(x, y, z)
