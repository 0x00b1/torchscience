import torch
from torch import Tensor


def jacobi_elliptic_sd(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function sd(u, m).

    Computes the Jacobi elliptic sd function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function sd is defined as:

    .. math::

       \text{sd}(u, m) = \frac{\text{sn}(u, m)}{\text{dn}(u, m)}

    where:
    - sn(u, m) is the Jacobi elliptic sine
    - dn(u, m) is the Jacobi elliptic delta amplitude

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - u: any real or complex value (the argument)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)
    - For m < 0 or m > 1, analytic continuation is used

    Special Values
    --------------
    - sd(0, m) = 0 for all m
    - sd(u, 0) = sin(u) (since sn(u,0) = sin(u) and dn(u,0) = 1)
    - sd(u, 1) = sinh(u) (since sn(u,1) = tanh(u) and dn(u,1) = sech(u))
    - sd(-u, m) = -sd(u, m) (odd function in u)

    Poles
    -----
    The sd function has poles where dn(u, m) = 0, which occurs at:
    u = (2n+1)K(m) + i*K'(m) for integer n

    Relationship to Other Jacobi Functions
    --------------------------------------
    The twelve Jacobi elliptic functions are formed from ratios of the
    three primary functions sn, cn, dn:

    .. math::

       \text{sd} = \frac{\text{sn}}{\text{dn}}, \quad
       \text{cd} = \frac{\text{cn}}{\text{dn}}, \quad
       \text{sc} = \frac{\text{sn}}{\text{cn}}

    Algorithm
    ---------
    The implementation computes sd(u, m) = sn(u, m) / dn(u, m) using
    the established implementations of the primary Jacobi functions.

    Applications
    ------------
    The Jacobi elliptic function sd appears in:
    - Elliptic filter design
    - Nonlinear wave equations
    - Conformal mapping problems
    - Integration of certain classes of algebraic functions

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
        The Jacobi elliptic function sd(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_sd(u, m)  # sd(0, m) = 0
    tensor([0.])

    Circular limit (m = 0):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_sd(u, m)  # equals sin(1.0) / 1 = sin(1.0)
    tensor([0.8415])

    Multiple values:

    >>> u = torch.tensor([0.0, 0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_sd(u, m)
    tensor([0.0000, 0.5157, 0.9109, 1.2836])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_sd(u, m)
    tensor([0.9031+0.3457j])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> sd = jacobi_elliptic_sd(u, m)
    >>> sd.backward()
    >>> u.grad
    tensor([0.4752])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The sd function has poles at u = (2n+1)K(m) + i*K'(m).
    - Near these poles, the function value may be very large or inf.

    See Also
    --------
    jacobi_elliptic_sn : Jacobi elliptic function sn(u, m)
    jacobi_elliptic_cn : Jacobi elliptic function cn(u, m)
    jacobi_elliptic_dn : Jacobi elliptic function dn(u, m)
    jacobi_elliptic_cd : Jacobi elliptic function cd(u, m)
    jacobi_elliptic_sc : Jacobi elliptic function sc(u, m)
    """
    return torch.ops.torchscience.jacobi_elliptic_sd(u, m)
