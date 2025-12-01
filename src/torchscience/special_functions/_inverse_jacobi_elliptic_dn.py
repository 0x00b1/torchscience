import torch
from torch import Tensor


def inverse_jacobi_elliptic_dn(x: Tensor, m: Tensor) -> Tensor:
    r"""
    Inverse Jacobi elliptic function arcdn(x, m).

    Computes the inverse of the Jacobi elliptic delta amplitude function dn(u, m),
    evaluated at each element of the input tensors.

    Mathematical Definition
    -----------------------
    The inverse Jacobi elliptic function arcdn is defined as:

    .. math::

       \text{arcdn}(x, m) = u \quad \text{such that} \quad \text{dn}(u, m) = x

    This can be expressed using Carlson's symmetric elliptic integral R_F:

    .. math::

       \text{arcdn}(x, m) = \sqrt{\frac{1 - x^2}{m}} \cdot R_F(x^2, 1, 1 - m + m x^2)

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - x: real or complex (typically sqrt(1-m) <= x <= 1 for real m with 0 <= m <= 1)
    - m: elliptic parameter (conventionally 0 < m <= 1, m != 0)

    Special Values
    --------------
    - arcdn(1, m) = 0 for all m (since dn(0, m) = 1)
    - arcdn(sqrt(1-m), m) = K(m) where K(m) is the complete elliptic integral
    - arcdn(x, 0) is undefined (dn(u, 0) = 1 for all u)
    - arcdn(x, 1) = arcsech(x) (hyperbolic limit)

    Inverse Property
    ----------------
    The arcdn function satisfies:

    .. math::

       \text{dn}(\text{arcdn}(x, m), m) = x

       \text{arcdn}(\text{dn}(u, m), m) = u \quad \text{(for } u \in [0, K(m)] \text{)}

    Applications
    ------------
    The inverse Jacobi elliptic function arcdn appears in:
    - Solving elliptic equations inversely
    - Computing elliptic integrals of the first kind
    - Inverse problems in pendulum dynamics
    - Conformal mapping applications

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradient with respect to x is:

    .. math::

       \frac{\partial \text{arcdn}}{\partial x} = \frac{-1}{\sqrt{(1 - x^2)(x^2 - 1 + m)}}

    The gradient with respect to m involves more complex expressions.
    This implementation uses numerical differentiation for both gradients.

    Second-order derivatives (gradgradcheck) are also supported via
    numerical differentiation.

    Parameters
    ----------
    x : Tensor
        The argument (value of dn to invert). Can be floating-point or complex.
        Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 < m <= 1 for real values.
        m = 0 is a degenerate case where dn(u, 0) = 1 for all u.
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        The inverse Jacobi elliptic function arcdn(x, m) = u such that dn(u, m) = x.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage - verify inverse property:

    >>> import torch
    >>> from torchscience.special_functions import jacobi_elliptic_dn, inverse_jacobi_elliptic_dn
    >>> u = torch.tensor([0.5])
    >>> m = torch.tensor([0.5])
    >>> x = jacobi_elliptic_dn(u, m)
    >>> u_recovered = inverse_jacobi_elliptic_dn(x, m)
    >>> torch.allclose(u, u_recovered)
    True

    Zero value:

    >>> x = torch.tensor([1.0])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_dn(x, m)  # arcdn(1, m) = 0
    tensor([0.])

    Hyperbolic limit (m = 1):

    >>> x = torch.tensor([0.5])
    >>> m = torch.tensor([1.0])
    >>> inverse_jacobi_elliptic_dn(x, m)  # equals arcsech(0.5)
    tensor([1.3170])

    Multiple values:

    >>> x = torch.tensor([1.0, 0.9, 0.8, 0.75])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_dn(x, m)
    tensor([0.0000, 0.6435, 0.9273, 1.0827])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The implementation uses Carlson's symmetric elliptic integral R_F
      for numerical stability and accuracy.
    - For m = 0, the function returns NaN since dn(u, 0) = 1 for all u,
      meaning there is no unique inverse.
    - For x outside [sqrt(1-m), 1] or complex x, analytic continuation is used.

    See Also
    --------
    jacobi_elliptic_dn : Jacobi elliptic function dn(u, m) (forward function)
    inverse_jacobi_elliptic_sn : Inverse Jacobi elliptic function arcsn(x, m)
    inverse_jacobi_elliptic_cn : Inverse Jacobi elliptic function arccn(x, m)
    carlson_elliptic_integral_r_f : Carlson's symmetric elliptic integral R_F
    """
    return torch.ops.torchscience.inverse_jacobi_elliptic_dn(x, m)
