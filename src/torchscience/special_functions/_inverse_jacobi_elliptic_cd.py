import torch
from torch import Tensor


def inverse_jacobi_elliptic_cd(x: Tensor, m: Tensor) -> Tensor:
    r"""
    Inverse Jacobi elliptic function arccd(x, m).

    Computes the inverse of the Jacobi elliptic cd function, finding u such
    that cd(u, m) = x.

    Mathematical Definition
    -----------------------
    The inverse Jacobi elliptic function arccd is defined as:

    .. math::

       \text{arccd}(x, m) = u \quad \text{such that} \quad \text{cd}(u, m) = x

    where cd(u, m) = cn(u, m) / dn(u, m) is the Jacobi elliptic cd function.

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - x: any real or complex value (the argument)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)
    - For m < 0 or m > 1, analytic continuation is used

    Special Values
    --------------
    - arccd(1, m) = 0 for all m (since cd(0, m) = 1)
    - arccd(x, 0) = arccos(x) (circular limit)
    - arccd(x, 1) is only defined for x = 1 (since cd(u, 1) = 1 for all u)

    Inverse Property
    ----------------
    For appropriate domains:

    .. math::

       \text{cd}(\text{arccd}(x, m), m) = x

       \text{arccd}(\text{cd}(u, m), m) = u

    Algorithm
    ---------
    The implementation uses Newton's method to solve cd(u, m) = x for u.
    The derivative d(cd)/du = -sn(u,m) * (1-m) / dn(u,m)^2 is used for
    Newton iteration.

    Applications
    ------------
    The inverse Jacobi elliptic function arccd appears in:
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
        The value of cd(u, m) for which to find u. Can be floating-point
        or complex. Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 <= m <= 1 for real values.
        Broadcasting with x is supported.

    Returns
    -------
    Tensor
        The inverse Jacobi elliptic function arccd(x, m), i.e., the value u
        such that cd(u, m) = x. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage - inverse property:

    >>> x = torch.tensor([1.0])
    >>> m = torch.tensor([0.5])
    >>> inverse_jacobi_elliptic_cd(x, m)  # arccd(1, m) = 0
    tensor([0.])

    Circular limit (m = 0):

    >>> x = torch.tensor([0.5])
    >>> m = torch.tensor([0.0])
    >>> inverse_jacobi_elliptic_cd(x, m)  # equals arccos(0.5)
    tensor([1.0472])

    Verifying inverse property:

    >>> from torchscience.special_functions import jacobi_elliptic_cd
    >>> u = torch.tensor([0.5])
    >>> m = torch.tensor([0.5])
    >>> cd_u = jacobi_elliptic_cd(u, m)
    >>> inverse_jacobi_elliptic_cd(cd_u, m)  # should return u
    tensor([0.5000])

    Complex input:

    >>> x = torch.tensor([0.8 + 0.1j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> inverse_jacobi_elliptic_cd(x, m)
    tensor([0.6253-0.1357j])

    Autograd:

    >>> x = torch.tensor([0.8], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> u = inverse_jacobi_elliptic_cd(x, m)
    >>> u.backward()
    >>> x.grad
    tensor([-1.6856])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - For m = 1, cd(u, 1) = 1 for all u, so arccd(x, 1) is only defined
      for x = 1 (returning 0) and is NaN otherwise.
    - The Newton iteration may not converge for all inputs, particularly
      near singularities of the cd function.
    - For numerical stability, the implementation includes damping for
      large Newton steps.

    See Also
    --------
    jacobi_elliptic_cd : Jacobi elliptic function cd(u, m)
    inverse_jacobi_elliptic_sd : Inverse Jacobi elliptic function arcsd(x, m)
    inverse_jacobi_elliptic_sc : Inverse Jacobi elliptic function arcsc(x, m)
    """
    return torch.ops.torchscience.inverse_jacobi_elliptic_cd(x, m)
