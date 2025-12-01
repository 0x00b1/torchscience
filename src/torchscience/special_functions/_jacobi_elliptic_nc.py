import torch
from torch import Tensor


def jacobi_elliptic_nc(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function nc(u, m).

    Computes the Jacobi elliptic function nc for each pair of corresponding
    elements in the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function nc is defined as the reciprocal of cn:

    .. math::

        \text{nc}(u, m) = \frac{1}{\text{cn}(u, m)}

    where cn(u, m) is the Jacobi elliptic function cosine amplitude.

    Domain
    ------
    - u: any real or complex number
    - m: elliptic parameter (0 <= m <= 1 for standard real case)
      - For m < 0 or m > 1, the function extends via analytic continuation

    Special Values
    --------------
    - nc(0, m) = 1 for all m
    - nc(u, 0) = sec(u) = 1/cos(u) for all u
    - nc(u, 1) = cosh(u)
    - nc has poles at zeros of cn(u, m)

    Periodicity
    -----------
    The function nc(u, m) is periodic in u with period 4K(m), where K(m)
    is the complete elliptic integral of the first kind.

    Relations to Other Jacobi Functions
    -----------------------------------
    The reciprocal Jacobi elliptic functions are:

    .. math::

        \text{nd}(u, m) &= \frac{1}{\text{dn}(u, m)} \\
        \text{nc}(u, m) &= \frac{1}{\text{cn}(u, m)} \\
        \text{ns}(u, m) &= \frac{1}{\text{sn}(u, m)}

    Algorithm
    ---------
    The implementation computes 1/cn(u, m) using the arithmetic-geometric
    mean (AGM) method for the underlying cn function.

    Applications
    ------------
    Jacobi elliptic functions appear in many physical and mathematical contexts:
    - Pendulum motion with large amplitudes
    - Nonlinear wave equations (solitons)
    - Conformal mapping in complex analysis
    - Elliptic curve cryptography
    - Signal processing and filter design

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are computed using numerical differentiation for stability.

    Second-order derivatives (gradgradcheck) are also supported using
    numerical differentiation.

    Parameters
    ----------
    u : Tensor
        The argument (amplitude). Can be any real or complex number.
        Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. For real computations, typically 0 <= m <= 1.
        Broadcasting with u is supported.

    Returns
    -------
    Tensor
        The Jacobi elliptic function nc(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage at u = 0:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_nc(u, m)
    tensor([1.0000])

    nc(u, 0) = sec(u) = 1/cos(u):

    >>> u = torch.tensor([0.0, 0.5, 1.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_nc(u, m)
    tensor([1.0000, 1.1395, 1.8508])
    >>> 1.0 / torch.cos(u)
    tensor([1.0000, 1.1395, 1.8508])

    nc(u, 1) = cosh(u):

    >>> u = torch.tensor([0.0, 1.0, 2.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_nc(u, m)
    tensor([1.0000, 1.5431, 3.7622])
    >>> torch.cosh(u)
    tensor([1.0000, 1.5431, 3.7622])

    Autograd:

    >>> u = torch.tensor([0.5], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> y = jacobi_elliptic_nc(u, m)
    >>> y.backward()
    >>> u.grad is not None
    True

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_nc(u, m)
    tensor([...])

    .. warning:: Poles

       The function nc has poles at zeros of cn(u, m), which occur at
       u = (2n+1)K(m) for integer n. Near these points, the function
       values can be very large or infinite.

    .. warning:: Numerical precision near m = 1

       Near m = 1, the function approaches cosh(u) which grows
       exponentially. For large |u|, this may overflow.

    Notes
    -----
    - The parameter m is sometimes called the "elliptic parameter" or
      "parameter". Some references use the modulus k where m = k^2.
    - For m = 0, the function reduces to sec(u) = 1/cos(u).
    - For m = 1, the function reduces to cosh(u).

    See Also
    --------
    jacobi_elliptic_cn : Jacobi elliptic function cn
    jacobi_elliptic_nd : Jacobi elliptic function nd = 1/dn
    jacobi_elliptic_ns : Jacobi elliptic function ns = 1/sn
    complete_legendre_elliptic_integral_k : Complete elliptic integral K(m)
    """
    return torch.ops.torchscience.jacobi_elliptic_nc(u, m)
