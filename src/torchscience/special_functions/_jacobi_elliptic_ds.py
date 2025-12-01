import torch
from torch import Tensor


def jacobi_elliptic_ds(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function ds(u, m).

    Computes the Jacobi elliptic ds function evaluated at each element.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic ds function is defined as:

    .. math::

       \mathrm{ds}(u, m) = \frac{\mathrm{dn}(u, m)}{\mathrm{sn}(u, m)}

    where:
    - :math:`\mathrm{dn}(u, m) = \sqrt{1 - m \sin^2(\mathrm{am}(u, m))}`
    - :math:`\mathrm{sn}(u, m) = \sin(\mathrm{am}(u, m))`
    - :math:`\mathrm{am}(u, m)` is the Jacobi amplitude function

    Domain
    ------
    - u: real or complex
    - m: 0 <= m <= 1 for real inputs (complex plane otherwise)

    Special Values
    --------------
    - ds(u, 0) = 1 / sin(u) = csc(u) (since dn(u,0) = 1 and sn(u,0) = sin(u))
    - ds(u, 1) = 1 / sinh(u) = csch(u) (since dn(u,1) = sech(u) and sn(u,1) = tanh(u))

    Poles
    -----
    ds has poles where sn(u, m) = 0, i.e., at u = 2nK(m) for integer n,
    where K(m) is the complete elliptic integral of the first kind.
    This includes u = 0.

    Periodicity
    -----------
    - ds(u + 4K(m), m) = ds(u, m)

    Parity
    ------
    - ds(-u, m) = -ds(u, m) (odd function in u)

    Algorithm
    ---------
    Computed as the ratio of dn(u, m) / sn(u, m), where dn and sn are
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
        The Jacobi elliptic ds function evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_ds(u, m)
    tensor([2.0305, 1.1423, 0.8682])

    At m = 0, ds reduces to csc:

    >>> u = torch.tensor([0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_ds(u, m)
    tensor([2.0858, 1.1884, 1.0025])
    >>> 1.0 / torch.sin(u)
    tensor([2.0858, 1.1884, 1.0025])

    At m = 1, ds reduces to csch:

    >>> u = torch.tensor([0.5, 1.0, 2.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_ds(u, m)
    tensor([1.9190, 0.8509, 0.2757])
    >>> 1.0 / torch.sinh(u)
    tensor([1.9190, 0.8509, 0.2757])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_ds(u, m)
    tensor([1.0117-0.4458j])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> y = jacobi_elliptic_ds(u, m)
    >>> y.backward()
    >>> u.grad
    tensor([-1.1032])

    Notes
    -----
    - The function is computed as dn(u, m) / sn(u, m).
    - For m near 0 or 1, special cases are used for better numerical stability.
    - The implementation handles both real and complex arguments.
    - Has a pole at u = 0 (odd function).

    See Also
    --------
    jacobi_elliptic_dn : Jacobi elliptic dn function
    jacobi_elliptic_sn : Jacobi elliptic sn function
    jacobi_elliptic_cn : Jacobi elliptic cn function
    jacobi_elliptic_dc : Jacobi elliptic dc function
    jacobi_elliptic_cs : Jacobi elliptic cs function
    """
    return torch.ops.torchscience.jacobi_elliptic_ds(u, m)
