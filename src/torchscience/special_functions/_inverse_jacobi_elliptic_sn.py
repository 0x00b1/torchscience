import torch
from torch import Tensor


def inverse_jacobi_elliptic_sn(x: Tensor, m: Tensor) -> Tensor:
    r"""
    Inverse Jacobi elliptic function arcsn(x, m).

    Computes the inverse of the Jacobi elliptic sine function sn(u, m),
    evaluated at each element of the input tensors.

    Mathematical Definition
    -----------------------
    The inverse Jacobi elliptic function arcsn is defined as:

    .. math::

       \text{arcsn}(x, m) = u \quad \text{such that} \quad \text{sn}(u, m) = x

    This can be expressed using Carlson's symmetric elliptic integral R_F:

    .. math::

       \text{arcsn}(x, m) = x \cdot R_F(1 - x^2, 1 - m x^2, 1)

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - x: real or complex (typically |x| <= 1 for real m with 0 <= m <= 1)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)

    Special Values
    --------------
    - arcsn(0, m) = 0 for all m
    - arcsn(1, m) = K(m) where K(m) is the complete elliptic integral
    - arcsn(x, 0) = arcsin(x) (circular limit)
    - arcsn(x, 1) = arctanh(x) (hyperbolic limit)

    Inverse Property
    ----------------
    The arcsn function satisfies:

    .. math::

       \text{sn}(\text{arcsn}(x, m), m) = x

       \text{arcsn}(\text{sn}(u, m), m) = u \quad \text{(for } u \in [0, K(m)] \text{)}

    Applications
    ------------
    The inverse Jacobi elliptic function arcsn appears in:
    - Solving elliptic equations inversely
    - Computing elliptic integrals of the first kind
    - Inverse problems in pendulum dynamics
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

       \frac{\partial \text{arcsn}}{\partial x} = \frac{1}{\sqrt{(1 - x^2)(1 - m x^2)}}

    The gradient with respect to m involves more complex expressions.
    This implementation uses numerical differentiation for both gradients.

    Second-order derivatives (gradgradcheck) are also supported via
    numerical differentiation.

    Parameters
    ----------
    x : Tensor
        The argument (value of sn to invert). Can be floating-point or complex.
        Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 <= m <= 1 for real values.
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        The inverse Jacobi elliptic function arcsn(x, m) = u such that sn(u, m) = x.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage - verify inverse property:

    >>> import torch
    >>> from torchscience.special_functions import jacobi_elliptic_sn, inverse_jacobi_elliptic_sn
    >>> u = torch.tensor([0.5])
    >>> m = torch.tensor([0.5])
    >>> x = jacobi_elliptic_sn(u, m)
    >>> u_recovered = inverse_jacobi_elliptic_sn(x, m)
    >>> torch.allclose(u, u_recovered)
    True

    Circular limit (m = 0):

    >>> x = torch.tensor([0.5])
    >>> m = torch.tensor([0.0])
    >>> inverse_jacobi_elliptic_sn(x, m)  # equals arcsin(0.5)
    tensor([0.5236])

    Hyperbolic limit (m = 1):

    >>> x = torch.tensor([0.5])
    >>> m = torch.tensor([1.0])
    >>> inverse_jacobi_elliptic_sn(x, m)  # equals arctanh(0.5)
    tensor([0.5493])

    Multiple values:

    >>> x = torch.tensor([0.0, 0.25, 0.5, 0.75])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_sn(x, m)
    tensor([0.0000, 0.2533, 0.5294, 0.8796])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The implementation uses Carlson's symmetric elliptic integral R_F
      for numerical stability and accuracy.
    - For x outside [-1, 1] or complex x, analytic continuation is used.

    See Also
    --------
    jacobi_elliptic_sn : Jacobi elliptic function sn(u, m) (forward function)
    inverse_jacobi_elliptic_cn : Inverse Jacobi elliptic function arccn(x, m)
    inverse_jacobi_elliptic_dn : Inverse Jacobi elliptic function arcdn(x, m)
    carlson_elliptic_integral_r_f : Carlson's symmetric elliptic integral R_F
    """
    return torch.ops.torchscience.inverse_jacobi_elliptic_sn(x, m)
