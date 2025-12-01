import torch
from torch import Tensor


def jacobi_elliptic_sc(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function sc(u, m).

    Computes the Jacobi elliptic sc function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function sc is defined as:

    .. math::

       \text{sc}(u, m) = \frac{\text{sn}(u, m)}{\text{cn}(u, m)}

    where:
    - sn(u, m) is the Jacobi elliptic sine
    - cn(u, m) is the Jacobi elliptic cosine

    This is the elliptic analog of the tangent function. In fact:

    .. math::

       \text{sc}(u, m) = \tan(\text{am}(u, m))

    where am(u, m) is the Jacobi amplitude function.

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - u: any real or complex value (the argument)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)
    - For m < 0 or m > 1, analytic continuation is used

    Special Values
    --------------
    - sc(0, m) = 0 for all m (since sn(0,m) = 0 and cn(0,m) = 1)
    - sc(u, 0) = tan(u) (since sn(u,0) = sin(u) and cn(u,0) = cos(u))
    - sc(u, 1) = sinh(u) (since sn(u,1) = tanh(u) and cn(u,1) = sech(u))
    - sc(-u, m) = -sc(u, m) (odd function in u)

    Poles
    -----
    The sc function has poles where cn(u, m) = 0, which occurs at:
    u = (2n+1)K(m) for integer n, where K(m) is the complete elliptic integral

    Relationship to Other Jacobi Functions
    --------------------------------------
    The twelve Jacobi elliptic functions are formed from ratios of the
    three primary functions sn, cn, dn:

    .. math::

       \text{sd} = \frac{\text{sn}}{\text{dn}}, \quad
       \text{cd} = \frac{\text{cn}}{\text{dn}}, \quad
       \text{sc} = \frac{\text{sn}}{\text{cn}}

    Identity:

    .. math::

       \text{sc}^2(u, m) + 1 = \text{nc}^2(u, m)

    where nc = 1/cn.

    Algorithm
    ---------
    The implementation computes sc(u, m) = sn(u, m) / cn(u, m) using
    the established implementations of the primary Jacobi functions.

    Applications
    ------------
    The Jacobi elliptic function sc appears in:
    - Elliptic filter design
    - Conformal mapping problems
    - Integration of trigonometric-like expressions with elliptic modulus
    - As the "elliptic tangent" in various physical models

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    This implementation uses numerical differentiation for gradients.

    Second-order derivatives (gradgradcheck) are also supported via
    numerical differentiation.

    Parameters
    ----------
    u : Tensor
        The argument (elliptic argument). Can be floating-point or complex.
        Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 <= m <= 1 for real values.
        Broadcasting with u is supported.

    Returns
    -------
    Tensor
        The Jacobi elliptic function sc(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_sc(u, m)  # sc(0, m) = 0
    tensor([0.])

    Circular limit (m = 0):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_sc(u, m)  # equals tan(1.0)
    tensor([1.5574])

    Hyperbolic limit (m = 1):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_sc(u, m)  # equals sinh(1.0)
    tensor([1.1752])

    Multiple values:

    >>> u = torch.tensor([0.0, 0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_sc(u, m)
    tensor([0.0000, 0.5604, 1.3355, 2.9437])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_sc(u, m)
    tensor([1.2356+0.6982j])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> sc = jacobi_elliptic_sc(u, m)
    >>> sc.backward()
    >>> u.grad
    tensor([1.5934])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The sc function has poles at u = (2n+1)K(m) (odd multiples of K).
    - Near these poles, the function value may be very large or inf.
    - sc is the elliptic generalization of tan; at m=0 it reduces to tan(u).

    See Also
    --------
    jacobi_elliptic_sn : Jacobi elliptic function sn(u, m)
    jacobi_elliptic_cn : Jacobi elliptic function cn(u, m)
    jacobi_elliptic_dn : Jacobi elliptic function dn(u, m)
    jacobi_elliptic_sd : Jacobi elliptic function sd(u, m)
    jacobi_elliptic_cd : Jacobi elliptic function cd(u, m)
    """
    return torch.ops.torchscience.jacobi_elliptic_sc(u, m)
