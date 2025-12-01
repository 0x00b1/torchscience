import torch
from torch import Tensor


def jacobi_elliptic_dc(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function dc(u, m).

    Computes the Jacobi elliptic dc function evaluated at each element.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic dc function is defined as:

    .. math::

       \mathrm{dc}(u, m) = \frac{\mathrm{dn}(u, m)}{\mathrm{cn}(u, m)}

    where:
    - :math:`\mathrm{dn}(u, m) = \sqrt{1 - m \sin^2(\mathrm{am}(u, m))}`
    - :math:`\mathrm{cn}(u, m) = \cos(\mathrm{am}(u, m))`
    - :math:`\mathrm{am}(u, m)` is the Jacobi amplitude function

    Domain
    ------
    - u: real or complex
    - m: 0 <= m <= 1 for real inputs (complex plane otherwise)

    Special Values
    --------------
    - dc(0, m) = 1 for all m (since dn(0,m) = 1 and cn(0,m) = 1)
    - dc(u, 0) = 1 / cos(u) = sec(u) (since dn(u,0) = 1 and cn(u,0) = cos(u))
    - dc(u, 1) = 1 (since dn(u,1) = cn(u,1) = sech(u))

    Poles
    -----
    dc has poles where cn(u, m) = 0, i.e., at u = (2n+1)K(m) for integer n,
    where K(m) is the complete elliptic integral of the first kind.

    Periodicity
    -----------
    - dc(u + 4K(m), m) = dc(u, m)

    Algorithm
    ---------
    Computed as the ratio of dn(u, m) / cn(u, m), where dn and cn are
    computed using the AGM method.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

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
        The Jacobi elliptic dc function evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0, 0.5, 1.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_dc(u, m)
    tensor([1.0000, 1.0695, 1.2305])

    dc(0, m) = 1 for all m:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.0, 0.5, 1.0])
    >>> jacobi_elliptic_dc(u, m)
    tensor([1., 1., 1.])

    At m = 0, dc reduces to sec:

    >>> u = torch.tensor([0.0, 0.5, 1.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_dc(u, m)
    tensor([1.0000, 1.1395, 1.8508])
    >>> 1.0 / torch.cos(u)
    tensor([1.0000, 1.1395, 1.8508])

    At m = 1, dc = 1:

    >>> u = torch.tensor([0.0, 1.0, 2.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_dc(u, m)
    tensor([1., 1., 1.])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_dc(u, m)
    tensor([1.1243+0.2887j])

    Autograd:

    >>> u = torch.tensor([0.5], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> y = jacobi_elliptic_dc(u, m)
    >>> y.backward()
    >>> u.grad
    tensor([0.5474])

    Notes
    -----
    - The function is computed as dn(u, m) / cn(u, m).
    - For m near 0 or 1, special cases are used for better numerical stability.
    - The implementation handles both real and complex arguments.

    See Also
    --------
    jacobi_elliptic_dn : Jacobi elliptic dn function
    jacobi_elliptic_cn : Jacobi elliptic cn function
    jacobi_elliptic_sn : Jacobi elliptic sn function
    jacobi_elliptic_ds : Jacobi elliptic ds function
    jacobi_elliptic_cs : Jacobi elliptic cs function
    """
    return torch.ops.torchscience.jacobi_elliptic_dc(u, m)
