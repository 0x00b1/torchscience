import torch
from torch import Tensor


def jacobi_elliptic_ns(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function ns(u, m).

    Computes the Jacobi elliptic function ns for each pair of corresponding
    elements in the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function ns is defined as the reciprocal of sn:

    .. math::

        \text{ns}(u, m) = \frac{1}{\text{sn}(u, m)}

    where sn(u, m) is the Jacobi elliptic function sine amplitude.

    Domain
    ------
    - u: any real or complex number (except zeros of sn)
    - m: elliptic parameter (0 <= m <= 1 for standard real case)
      - For m < 0 or m > 1, the function extends via analytic continuation

    Special Values
    --------------
    - ns(0, m) has a pole (sn(0, m) = 0)
    - ns(u, 0) = csc(u) = 1/sin(u) for all u
    - ns(u, 1) = coth(u)
    - ns(K(m), m) = 1, where K(m) is the complete elliptic integral
      of the first kind

    Periodicity
    -----------
    The function ns(u, m) is periodic in u with period 4K(m), where K(m)
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
    The implementation computes 1/sn(u, m) using the arithmetic-geometric
    mean (AGM) method for the underlying sn function.

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
        The Jacobi elliptic function ns(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    ns(u, 0) = csc(u) = 1/sin(u):

    >>> u = torch.tensor([0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_ns(u, m)
    tensor([2.0858, 1.1884, 1.0025])
    >>> 1.0 / torch.sin(u)
    tensor([2.0858, 1.1884, 1.0025])

    ns(u, 1) = coth(u):

    >>> u = torch.tensor([0.5, 1.0, 2.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_ns(u, m)
    tensor([2.1640, 1.3130, 1.0373])
    >>> 1.0 / torch.tanh(u)
    tensor([2.1640, 1.3130, 1.0373])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> y = jacobi_elliptic_ns(u, m)
    >>> y.backward()
    >>> u.grad is not None
    True

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_ns(u, m)
    tensor([...])

    .. warning:: Poles

       The function ns has poles at zeros of sn(u, m), which occur at
       u = 2nK(m) for integer n. In particular, ns(0, m) = infinity.
       Near these points, the function values can be very large or infinite.

    .. warning:: Numerical precision near m = 1

       Near m = 1, the function approaches coth(u). For small |u|,
       coth(u) grows without bound.

    Notes
    -----
    - The parameter m is sometimes called the "elliptic parameter" or
      "parameter". Some references use the modulus k where m = k^2.
    - For m = 0, the function reduces to csc(u) = 1/sin(u).
    - For m = 1, the function reduces to coth(u).

    See Also
    --------
    jacobi_elliptic_sn : Jacobi elliptic function sn
    jacobi_elliptic_nd : Jacobi elliptic function nd = 1/dn
    jacobi_elliptic_nc : Jacobi elliptic function nc = 1/cn
    complete_legendre_elliptic_integral_k : Complete elliptic integral K(m)
    """
    return torch.ops.torchscience.jacobi_elliptic_ns(u, m)
