import torch
from torch import Tensor


def jacobi_elliptic_cd(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function cd(u, m).

    Computes the Jacobi elliptic cd function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function cd is defined as:

    .. math::

       \text{cd}(u, m) = \frac{\text{cn}(u, m)}{\text{dn}(u, m)}

    where:
    - cn(u, m) is the Jacobi elliptic cosine
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
    - cd(0, m) = 1 for all m (since cn(0,m) = 1 and dn(0,m) = 1)
    - cd(u, 0) = cos(u) (since cn(u,0) = cos(u) and dn(u,0) = 1)
    - cd(u, 1) = 1 (since cn(u,1) = sech(u) and dn(u,1) = sech(u))
    - cd(-u, m) = cd(u, m) (even function in u)

    Poles
    -----
    The cd function has poles where dn(u, m) = 0, which occurs at:
    u = (2n+1)K(m) + i*K'(m) for integer n

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

       \text{cd}^2(u, m) + m \cdot \text{sd}^2(u, m) = 1

    Algorithm
    ---------
    The implementation computes cd(u, m) = cn(u, m) / dn(u, m) using
    the established implementations of the primary Jacobi functions.

    Applications
    ------------
    The Jacobi elliptic function cd appears in:
    - Elliptic filter design (particularly Chebyshev and Cauer filters)
    - Conformal mapping of rectangles to circular domains
    - Solutions of certain nonlinear differential equations
    - Parametrization of elliptic curves

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
        The Jacobi elliptic function cd(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_cd(u, m)  # cd(0, m) = 1
    tensor([1.])

    Circular limit (m = 0):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_cd(u, m)  # equals cos(1.0) / 1 = cos(1.0)
    tensor([0.5403])

    Hyperbolic limit (m = 1):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_cd(u, m)  # equals 1
    tensor([1.])

    Multiple values:

    >>> u = torch.tensor([0.0, 0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_cd(u, m)
    tensor([1.0000, 0.9206, 0.6820, 0.3291])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_cd(u, m)
    tensor([0.6515-0.2125j])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> cd = jacobi_elliptic_cd(u, m)
    >>> cd.backward()
    >>> u.grad
    tensor([-0.6218])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The cd function has poles at u = (2n+1)K(m) + i*K'(m).
    - Near these poles, the function value may be very large or inf.
    - At m = 1, cd(u, 1) = 1 for all u (constant function).

    See Also
    --------
    jacobi_elliptic_sn : Jacobi elliptic function sn(u, m)
    jacobi_elliptic_cn : Jacobi elliptic function cn(u, m)
    jacobi_elliptic_dn : Jacobi elliptic function dn(u, m)
    jacobi_elliptic_sd : Jacobi elliptic function sd(u, m)
    jacobi_elliptic_sc : Jacobi elliptic function sc(u, m)
    """
    return torch.ops.torchscience.jacobi_elliptic_cd(u, m)
