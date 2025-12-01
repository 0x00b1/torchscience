import torch
from torch import Tensor


def incomplete_legendre_elliptic_integral_e(phi: Tensor, m: Tensor) -> Tensor:
    r"""
    Incomplete elliptic integral of the second kind E(phi, m).

    Computes the incomplete elliptic integral of the second kind evaluated at
    each pair of elements from the input tensors.

    Mathematical Definition
    -----------------------
    The incomplete elliptic integral of the second kind is defined as:

    .. math::

       E(\phi, m) = \int_0^{\phi} \sqrt{1 - m \sin^2\theta} \, d\theta

    This is related to Carlson's symmetric elliptic integrals by:

    .. math::

       E(\phi, m) = \sin(\phi) \cdot R_F(\cos^2\phi, 1-m\sin^2\phi, 1)
                    - \frac{m}{3} \sin^3(\phi) \cdot R_D(\cos^2\phi, 1-m\sin^2\phi, 1)

    where R_F and R_D are Carlson's symmetric elliptic integrals.

    Special Values
    --------------
    - E(0, m) = 0 for any m
    - E(pi/2, m) = complete E(m)

    Domain
    ------
    - For real phi: Any real value
    - For real m: m <= 1 for real results (for m > 1, result is complex)
    - For complex inputs: Defined on the entire complex plane

    Applications
    ------------
    The incomplete elliptic integral of the second kind appears in:
    - Arc length of an ellipse from a starting point
    - Geodesics on an ellipsoid
    - Pendulum motion (arbitrary amplitude)
    - Electromagnetic field calculations
    - Various problems in potential theory

    Parameter Convention
    --------------------
    This function uses the parameter m (the "parameter" convention), where
    m = k^2 and k is the elliptic modulus. Some references use k directly.

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types
    - Inputs are broadcast to a common shape

    Autograd Support
    ----------------
    Gradients are fully supported when phi.requires_grad or m.requires_grad
    is True.

    The analytical gradient with respect to phi is:

    .. math::

       \frac{\partial E}{\partial \phi} = \sqrt{1 - m \sin^2\phi}

    The gradient with respect to m is:

    .. math::

       \frac{\partial E}{\partial m} = \frac{E(\phi, m) - F(\phi, m)}{2m}

    where F(phi, m) is the incomplete elliptic integral of the first kind.

    Second-order derivatives are also supported.

    Parameters
    ----------
    phi : Tensor
        Input tensor for the amplitude (in radians). Can be floating-point
        or complex.
    m : Tensor
        Input tensor for the parameter. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The incomplete elliptic integral of the second kind E(phi, m) evaluated
        at each pair of elements. Output dtype matches the promoted input dtype.

    Examples
    --------
    Basic usage:

    >>> phi = torch.tensor([0.0, 0.5, 1.0, 1.5707963267948966])  # 0, 0.5, 1.0, pi/2
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_e(phi, m)
    tensor([0.0000, 0.4873, 0.9273, 1.3506])

    The value at phi=0 is always 0:

    >>> phi = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_e(phi, m)
    tensor([0.])

    At phi=pi/2, this equals the complete elliptic integral E(m):

    >>> import math
    >>> phi = torch.tensor([math.pi / 2])
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_e(phi, m)
    tensor([1.3506])  # Same as complete E(0.5)

    Complex input:

    >>> phi = torch.tensor([1.0 + 0.1j])
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_e(phi, m)
    tensor([0.9262+0.0890j])

    Autograd:

    >>> phi = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5], requires_grad=True)
    >>> y = incomplete_legendre_elliptic_integral_e(phi, m)
    >>> y.backward()
    >>> phi.grad
    tensor([0.8394])  # sqrt(1 - 0.5 * sin^2(1))
    >>> m.grad
    tensor([-0.0772])

    Notes
    -----
    - The implementation uses Carlson's symmetric forms R_F and R_D for
      numerical stability and accuracy.
    - Broadcasting is applied to phi and m.

    See Also
    --------
    complete_legendre_elliptic_integral_e : Complete elliptic integral E(m)
    carlson_elliptic_integral_r_f : Carlson's symmetric elliptic integral R_F
    carlson_elliptic_integral_r_d : Carlson's symmetric elliptic integral R_D
    """
    return torch.ops.torchscience.incomplete_legendre_elliptic_integral_e(
        phi, m
    )
