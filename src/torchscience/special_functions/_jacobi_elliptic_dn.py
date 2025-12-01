import torch
from torch import Tensor


def jacobi_elliptic_dn(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function dn(u, m).

    Computes the Jacobi elliptic function dn (delta amplitude) for each
    pair of corresponding elements in the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function dn is defined as:

    .. math::

        \text{dn}(u, m) = \sqrt{1 - m \cdot \text{sn}^2(u, m)}

    where sn(u, m) is the Jacobi elliptic sine function, and m is the
    elliptic parameter. Equivalently:

    .. math::

        \text{dn}(u, m) = \sqrt{1 - m \sin^2(\text{am}(u, m))}

    where am(u, m) is the Jacobi amplitude function.

    Domain
    ------
    - u: any real or complex number
    - m: elliptic parameter (0 <= m <= 1 for standard real case)
      - For m < 0 or m > 1, the function extends via analytic continuation

    Special Values
    --------------
    - dn(0, m) = 1 for all m
    - dn(u, 0) = 1 for all u
    - dn(u, 1) = sech(u) = 1/cosh(u)
    - dn(K(m), m) = sqrt(1 - m), where K(m) is the complete elliptic
      integral of the first kind

    Periodicity
    -----------
    The function dn(u, m) is periodic in u with period 2K(m), where K(m)
    is the complete elliptic integral of the first kind.

    Relations to Other Jacobi Functions
    -----------------------------------
    The three main Jacobi elliptic functions satisfy:

    .. math::

        \text{sn}^2(u, m) + \text{cn}^2(u, m) &= 1 \\
        \text{dn}^2(u, m) + m \cdot \text{sn}^2(u, m) &= 1 \\
        \text{dn}^2(u, m) - m \cdot \text{cn}^2(u, m) &= 1 - m

    Algorithm
    ---------
    The implementation uses the arithmetic-geometric mean (AGM) method
    to compute the Jacobi amplitude am(u, m), then computes
    dn = sqrt(1 - m * sin^2(am)).

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
    The gradients are computed analytically:

    .. math::

        \frac{\partial \text{dn}}{\partial u} &= -m \cdot \text{sn}(u, m) \cdot \text{cn}(u, m) \\
        \frac{\partial \text{dn}}{\partial m} &= -\frac{\text{sn}^2(u, m)}{2 \cdot \text{dn}(u, m)}

    Second-order derivatives (gradgradcheck) are also supported using
    numerical differentiation for stability.

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
        The Jacobi elliptic function dn(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage at u = 0:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_dn(u, m)
    tensor([1.0000])

    dn(u, 0) = 1 for all u:

    >>> u = torch.tensor([0.0, 1.0, 2.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_dn(u, m)
    tensor([1., 1., 1.])

    dn(u, 1) = sech(u):

    >>> u = torch.tensor([0.0, 1.0, 2.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_dn(u, m)
    tensor([1.0000, 0.6481, 0.2658])
    >>> 1.0 / torch.cosh(u)
    tensor([1.0000, 0.6481, 0.2658])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> y = jacobi_elliptic_dn(u, m)
    >>> y.backward()
    >>> u.grad
    tensor([-0.2872])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_dn(u, m)
    tensor([0.9110+0.0855j])

    .. warning:: Branch cuts for complex arguments

       For complex arguments, the function has branch cuts that depend on
       the values of both u and m. Care should be taken near these branch
       cuts where the function may be discontinuous.

    .. warning:: Numerical precision near m = 1

       Near m = 1, the function approaches sech(u) which decays
       exponentially. For large |u|, this may underflow to zero.

    Notes
    -----
    - The parameter m is sometimes called the "elliptic parameter" or
      "parameter". Some references use the modulus k where m = k^2.
    - For very small m, the function approaches 1.
    - For m = 1, the function reduces to sech(u) = 1/cosh(u).

    See Also
    --------
    jacobi_amplitude_am : Jacobi amplitude function
    complete_legendre_elliptic_integral_k : Complete elliptic integral K(m)
    """
    return torch.ops.torchscience.jacobi_elliptic_dn(u, m)
