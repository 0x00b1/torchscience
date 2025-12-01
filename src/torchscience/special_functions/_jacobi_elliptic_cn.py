import torch
from torch import Tensor


def jacobi_elliptic_cn(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function cn(u, m).

    Computes the Jacobi elliptic cn function evaluated at each element.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic cn function is defined as:

    .. math::

       \mathrm{cn}(u, m) = \cos(\mathrm{am}(u, m))

    where :math:`\mathrm{am}(u, m)` is the Jacobi amplitude function, which is
    the inverse of the incomplete elliptic integral of the first kind:

    .. math::

       u = F(\mathrm{am}, m) = \int_0^{\mathrm{am}} \frac{d\theta}{\sqrt{1 - m \sin^2(\theta)}}

    The parameter m is the "parameter convention" where m = k^2 (k is the modulus).

    Domain
    ------
    - u: real or complex
    - m: 0 <= m <= 1 for real inputs (complex plane otherwise)

    Special Values
    --------------
    - cn(0, m) = 1 for all m
    - cn(K(m), m) = 0 where K(m) is the complete elliptic integral of the first kind
    - cn(u, 0) = cos(u)
    - cn(u, 1) = sech(u) = 1/cosh(u)

    Periodicity
    -----------
    - cn(u + 4K(m), m) = cn(u, m) where K(m) is the complete elliptic integral

    Algorithm
    ---------
    Uses the arithmetic-geometric mean (AGM) descending Landen transformation
    to compute cn(u, m) efficiently and accurately.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Relationships with Other Jacobi Functions
    -----------------------------------------
    The Jacobi elliptic functions sn, cn, and dn satisfy:

    .. math::

       \mathrm{sn}^2(u, m) + \mathrm{cn}^2(u, m) &= 1 \\
       \mathrm{dn}^2(u, m) + m \cdot \mathrm{sn}^2(u, m) &= 1 \\
       \mathrm{cn}^2(u, m) + (1-m) &= \mathrm{dn}^2(u, m) + m \cdot \mathrm{cn}^2(u, m)

    Applications
    ------------
    Jacobi elliptic functions appear in:
    - Nonlinear wave equations (KdV, sine-Gordon)
    - Pendulum motion (exact solutions)
    - Conformal mapping
    - Elliptic filter design
    - Soliton solutions in integrable systems

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradient with respect to u is:

    .. math::

       \frac{\partial \mathrm{cn}}{\partial u} = -\mathrm{sn}(u, m) \cdot \mathrm{dn}(u, m)

    where sn and dn are the other Jacobi elliptic functions.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    u : Tensor
        The argument tensor. Broadcasting with m is supported.
    m : Tensor
        The parameter tensor. Must satisfy 0 <= m <= 1 for real inputs.
        Broadcasting with u is supported.

    Returns
    -------
    Tensor
        The Jacobi elliptic cn function evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0, 0.5, 1.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_cn(u, m)
    tensor([1.0000, 0.8758, 0.5959])

    cn(0, m) = 1 for all m:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.0, 0.5, 1.0])
    >>> jacobi_elliptic_cn(u, m)
    tensor([1., 1., 1.])

    At m = 0, cn reduces to cos:

    >>> u = torch.tensor([0.0, 1.0, 2.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_cn(u, m)
    tensor([ 1.0000,  0.5403, -0.4161])
    >>> torch.cos(u)
    tensor([ 1.0000,  0.5403, -0.4161])

    At m = 1, cn reduces to sech:

    >>> u = torch.tensor([0.0, 1.0, 2.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_cn(u, m)
    tensor([1.0000, 0.6481, 0.2658])
    >>> 1.0 / torch.cosh(u)
    tensor([1.0000, 0.6481, 0.2658])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_cn(u, m)
    tensor([0.8476-0.2175j])

    Autograd:

    >>> u = torch.tensor([0.5], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> y = jacobi_elliptic_cn(u, m)
    >>> y.backward()
    >>> u.grad
    tensor([-0.4501])

    Notes
    -----
    - The function is computed using the AGM (Arithmetic-Geometric Mean) method
      with descending Landen transformations.
    - For m near 0 or 1, special cases are used for better numerical stability.
    - The implementation handles both real and complex arguments.

    See Also
    --------
    jacobi_elliptic_sn : Jacobi elliptic sn function
    jacobi_elliptic_dn : Jacobi elliptic dn function
    jacobi_amplitude_am : Jacobi amplitude function
    complete_legendre_elliptic_integral_k : Complete elliptic integral of the first kind
    """
    return torch.ops.torchscience.jacobi_elliptic_cn(u, m)
