import torch
from torch import Tensor


def inverse_jacobi_elliptic_cn(x: Tensor, m: Tensor) -> Tensor:
    r"""
    Inverse Jacobi elliptic function arccn(x, m).

    Computes the inverse of the Jacobi elliptic cosine function cn(u, m),
    evaluated at each element of the input tensors.

    Mathematical Definition
    -----------------------
    The inverse Jacobi elliptic function arccn is defined as:

    .. math::

       \text{arccn}(x, m) = u \quad \text{such that} \quad \text{cn}(u, m) = x

    This can be expressed using Carlson's symmetric elliptic integral R_F:

    .. math::

       \text{arccn}(x, m) = \sqrt{1 - x^2} \cdot R_F(x^2, 1 - m + m x^2, 1)

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - x: real or complex (typically |x| <= 1 for real m with 0 <= m <= 1)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)

    Special Values
    --------------
    - arccn(1, m) = 0 for all m (since cn(0, m) = 1)
    - arccn(0, m) = K(m) where K(m) is the complete elliptic integral
    - arccn(x, 0) = arccos(x) (circular limit)
    - arccn(x, 1) = arcsech(x) (hyperbolic limit)

    Inverse Property
    ----------------
    The arccn function satisfies:

    .. math::

       \text{cn}(\text{arccn}(x, m), m) = x

       \text{arccn}(\text{cn}(u, m), m) = u \quad \text{(for } u \in [0, K(m)] \text{)}

    Applications
    ------------
    The inverse Jacobi elliptic function arccn appears in:
    - Solving elliptic equations inversely
    - Computing elliptic integrals of the first kind
    - Inverse problems in mechanical systems
    - Signal processing with elliptic filters

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

       \frac{\partial \text{arccn}}{\partial x} = \frac{-1}{\sqrt{(1 - x^2)(1 - m + m x^2)}}

    The gradient with respect to m involves more complex expressions.
    This implementation uses numerical differentiation for both gradients.

    Second-order derivatives (gradgradcheck) are also supported via
    numerical differentiation.

    Parameters
    ----------
    x : Tensor
        The argument (value of cn to invert). Can be floating-point or complex.
        Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 <= m <= 1 for real values.
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        The inverse Jacobi elliptic function arccn(x, m) = u such that cn(u, m) = x.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage - verify inverse property:

    >>> import torch
    >>> from torchscience.special_functions import jacobi_elliptic_cn, inverse_jacobi_elliptic_cn
    >>> u = torch.tensor([0.5])
    >>> m = torch.tensor([0.5])
    >>> x = jacobi_elliptic_cn(u, m)
    >>> u_recovered = inverse_jacobi_elliptic_cn(x, m)
    >>> torch.allclose(u, u_recovered)
    True

    Circular limit (m = 0):

    >>> x = torch.tensor([0.5])
    >>> m = torch.tensor([0.0])
    >>> inverse_jacobi_elliptic_cn(x, m)  # equals arccos(0.5)
    tensor([1.0472])

    Zero value:

    >>> x = torch.tensor([1.0])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_cn(x, m)  # arccn(1, m) = 0
    tensor([0.])

    Multiple values:

    >>> x = torch.tensor([1.0, 0.75, 0.5, 0.25])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_cn(x, m)
    tensor([0.0000, 0.7071, 1.1107, 1.4142])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The implementation uses Carlson's symmetric elliptic integral R_F
      for numerical stability and accuracy.
    - For x outside [-1, 1] or complex x, analytic continuation is used.

    See Also
    --------
    jacobi_elliptic_cn : Jacobi elliptic function cn(u, m) (forward function)
    inverse_jacobi_elliptic_sn : Inverse Jacobi elliptic function arcsn(x, m)
    inverse_jacobi_elliptic_dn : Inverse Jacobi elliptic function arcdn(x, m)
    carlson_elliptic_integral_r_f : Carlson's symmetric elliptic integral R_F
    """
    return torch.ops.torchscience.inverse_jacobi_elliptic_cn(x, m)
