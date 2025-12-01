import torch
from torch import Tensor


def jacobi_amplitude_am(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi amplitude function am(u, m).

    Computes the Jacobi amplitude function, which is the inverse of the
    incomplete elliptic integral of the first kind F(phi, m).

    Mathematical Definition
    -----------------------
    The Jacobi amplitude function is defined implicitly by:

    .. math::

       \text{am}(u, m) = \phi \quad \text{where} \quad u = F(\phi, m)

    where F is the incomplete elliptic integral of the first kind:

    .. math::

       F(\phi, m) = \int_0^{\phi} \frac{d\theta}{\sqrt{1 - m \sin^2\theta}}

    The amplitude function satisfies the fundamental relations with
    the Jacobi elliptic functions:

    .. math::

       \text{sn}(u, m) &= \sin(\text{am}(u, m)) \\
       \text{cn}(u, m) &= \cos(\text{am}(u, m)) \\
       \text{dn}(u, m) &= \sqrt{1 - m \sin^2(\text{am}(u, m))}

    Domain
    ------
    - u: any real or complex number
    - m: the parameter (0 <= m <= 1 for standard applications)
    - For m < 0 or m > 1, analytic continuation applies

    Special Values
    --------------
    - am(0, m) = 0 for all m
    - am(K(m), m) = pi/2 where K(m) is the complete elliptic integral of the first kind
    - am(u, 0) = u (reduces to identity when m = 0)
    - am(u, 1) = gd(u) = 2*arctan(exp(u)) - pi/2 (Gudermannian function)

    Algorithm
    ---------
    The implementation uses the Landen descending transformation, which
    converges rapidly for all parameter values:

    1. Compute the arithmetic-geometric mean (AGM) sequence starting from
       a_0 = 1, b_0 = sqrt(1 - m), c_0 = sqrt(m)
    2. Iterate: a_{n+1} = (a_n + b_n)/2, b_{n+1} = sqrt(a_n * b_n),
       c_{n+1} = (a_n - b_n)/2
    3. Stop when c_n is sufficiently small
    4. Back-substitute to find the amplitude

    This algorithm provides:
    - Consistent results across CPU and CUDA devices
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Jacobi amplitude function appears in:
    - Pendulum problems (exact solution for arbitrary amplitudes)
    - Conformal mapping and complex analysis
    - Elliptic curve cryptography
    - Nonlinear wave equations (KdV, sine-Gordon)
    - Statistical mechanics and quantum field theory

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are computed using:

    .. math::

       \frac{\partial \text{am}}{\partial u} = \text{dn}(u, m)
                                             = \sqrt{1 - m \sin^2(\text{am}(u, m))}

    The gradient with respect to m is computed using numerical differentiation
    for stability across the parameter range.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    u : Tensor
        The argument tensor. For the standard case where u = F(phi, m),
        this represents the value of the incomplete elliptic integral.
        Broadcasting with m is supported.
    m : Tensor
        The parameter tensor. Standard range is 0 <= m <= 1.
        m = 0 gives the identity function, m = 1 gives the Gudermannian.
        Broadcasting with u is supported.

    Returns
    -------
    Tensor
        The Jacobi amplitude am(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_amplitude_am(u, m)  # am(0, m) = 0
    tensor([0.])

    Relationship with complete elliptic integral K(m):

    >>> from torchscience.special_functions import complete_legendre_elliptic_integral_k
    >>> m = torch.tensor([0.5])
    >>> K = complete_legendre_elliptic_integral_k(m)
    >>> jacobi_amplitude_am(K, m)  # Should be approximately pi/2
    tensor([1.5708])

    Special case m = 0 (identity function):

    >>> u = torch.tensor([0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.0])
    >>> jacobi_amplitude_am(u, m)  # am(u, 0) = u
    tensor([0.5000, 1.0000, 1.5000])

    Special case m = 1 (Gudermannian function):

    >>> u = torch.tensor([0.0, 0.5, 1.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_amplitude_am(u, m)  # gd(u) = 2*atan(exp(u)) - pi/2
    tensor([0.0000, 0.4804, 0.8657])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_amplitude_am(u, m)
    tensor([0.9732+0.4016j])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> phi = jacobi_amplitude_am(u, m)
    >>> phi.backward()
    >>> u.grad  # Should be dn(1.0, 0.5)
    tensor([0.8231])

    .. warning:: Parameter range

       While the function is defined for all real m, the standard
       applications use 0 <= m <= 1. For m < 0 or m > 1, the function
       involves complex values even for real u.

    .. warning:: Periodicity

       The amplitude function has periodicity properties related to
       the complete elliptic integral K(m):
       am(u + 2K, m) = am(u, m) + pi

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The Landen transformation provides excellent numerical stability
      and convergence across all parameter values.
    - For very large |u|, the periodicity property should be used to
      reduce u modulo 2K(m) before computation.

    See Also
    --------
    complete_legendre_elliptic_integral_k : Complete elliptic integral of the first kind K(m)
    incomplete_legendre_elliptic_integral_e : Incomplete elliptic integral of the second kind
    """
    return torch.ops.torchscience.jacobi_amplitude_am(u, m)
