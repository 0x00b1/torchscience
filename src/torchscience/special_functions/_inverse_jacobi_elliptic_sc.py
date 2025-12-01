import torch
from torch import Tensor


def inverse_jacobi_elliptic_sc(x: Tensor, m: Tensor) -> Tensor:
    r"""
    Inverse Jacobi elliptic function arcsc(x, m).

    Computes the inverse of the Jacobi elliptic sc function, finding u such
    that sc(u, m) = x.

    Mathematical Definition
    -----------------------
    The inverse Jacobi elliptic function arcsc is defined as:

    .. math::

       \text{arcsc}(x, m) = u \quad \text{such that} \quad \text{sc}(u, m) = x

    where sc(u, m) = sn(u, m) / cn(u, m) is the Jacobi elliptic sc function.

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - x: any real or complex value (the argument)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)
    - For m < 0 or m > 1, analytic continuation is used

    Special Values
    --------------
    - arcsc(0, m) = 0 for all m (since sc(0, m) = 0)
    - arcsc(x, 0) = arctan(x) (circular limit, since sc(u,0) = tan(u))
    - arcsc(x, 1) = arcsinh(x) (hyperbolic limit, since sc(u,1) = sinh(u))

    Inverse Property
    ----------------
    For appropriate domains:

    .. math::

       \text{sc}(\text{arcsc}(x, m), m) = x

       \text{arcsc}(\text{sc}(u, m), m) = u

    Algorithm
    ---------
    The implementation uses Newton's method to solve sc(u, m) = x for u.
    The derivative d(sc)/du = dn(u,m) / cn(u,m)^2 is used for Newton iteration.

    Applications
    ------------
    The inverse Jacobi elliptic function arcsc appears in:
    - Elliptic filter design (computing argument from function value)
    - Solving nonlinear wave equations
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
    x : Tensor
        The value of sc(u, m) for which to find u. Can be floating-point
        or complex. Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 <= m <= 1 for real values.
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        The inverse Jacobi elliptic function arcsc(x, m), i.e., the value u
        such that sc(u, m) = x. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage - inverse property:

    >>> x = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_sc(x, m)  # arcsc(0, m) = 0
    tensor([0.])

    Circular limit (m = 0):

    >>> x = torch.tensor([1.0])
    >>> m = torch.tensor([0.0])
    >>> inverse_jacobi_elliptic_sc(x, m)  # equals arctan(1.0) = pi/4
    tensor([0.7854])

    Hyperbolic limit (m = 1):

    >>> x = torch.tensor([1.0])
    >>> m = torch.tensor([1.0])
    >>> inverse_jacobi_elliptic_sc(x, m)  # equals arcsinh(1.0)
    tensor([0.8814])

    Verifying inverse property:

    >>> from torchscience.special_functions import jacobi_elliptic_sc
    >>> u = torch.tensor([0.5])
    >>> m = torch.tensor([0.5])
    >>> sc_u = jacobi_elliptic_sc(u, m)
    >>> inverse_jacobi_elliptic_sc(sc_u, m)  # should return u
    tensor([0.5000])

    Complex input:

    >>> x = torch.tensor([0.5 + 0.1j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> inverse_jacobi_elliptic_sc(x, m)
    tensor([0.4748+0.1021j])

    Autograd:

    >>> x = torch.tensor([0.5], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> u = inverse_jacobi_elliptic_sc(x, m)
    >>> u.backward()
    >>> x.grad
    tensor([0.8045])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The sc function has poles at u = (2n+1)K(m) where K(m) is the complete
      elliptic integral of the first kind. The inverse function may have
      branch cuts near these values.
    - The Newton iteration may not converge for all inputs, particularly
      near singularities of the sc function.
    - For numerical stability, the implementation includes damping for
      large Newton steps.

    See Also
    --------
    jacobi_elliptic_sc : Jacobi elliptic function sc(u, m)
    inverse_jacobi_elliptic_sd : Inverse Jacobi elliptic function arcsd(x, m)
    inverse_jacobi_elliptic_cd : Inverse Jacobi elliptic function arccd(x, m)
    """
    return torch.ops.torchscience.inverse_jacobi_elliptic_sc(x, m)
