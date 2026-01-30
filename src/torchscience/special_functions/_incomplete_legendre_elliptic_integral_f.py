import torch
from torch import Tensor


def incomplete_legendre_elliptic_integral_f(phi: Tensor, m: Tensor) -> Tensor:
    r"""
    Incomplete elliptic integral of the first kind F(phi, m).

    Computes the incomplete elliptic integral of the first kind evaluated at
    each pair of elements from the input tensors.

    Mathematical Definition
    -----------------------
    The incomplete elliptic integral of the first kind is defined as:

    .. math::

       F(\phi, m) = \int_0^{\phi} \frac{d\theta}{\sqrt{1 - m \sin^2\theta}}

    This is related to Carlson's symmetric elliptic integral by:

    .. math::

       F(\phi, m) = \sin(\phi) \cdot R_F(\cos^2\phi, 1-m\sin^2\phi, 1)

    where R_F is Carlson's symmetric elliptic integral of the first kind.

    Special Values
    --------------
    - F(0, m) = 0 for any m
    - F(pi/2, m) = K(m), the complete elliptic integral of the first kind
    - F(-phi, m) = -F(phi, m) (odd in phi)
    - F(phi, 0) = phi
    - F(phi, 1) = arctanh(sin(phi)) for 0 <= phi < pi/2

    Domain
    ------
    - For real phi: Any real value
    - For real m: 0 <= m <= 1 for real results (for m > 1, result may be complex)
    - For complex inputs: Defined on the entire complex plane

    Applications
    ------------
    The incomplete elliptic integral of the first kind appears in:
    - Pendulum period calculation (exact formula)
    - Geodesics on an ellipsoid
    - Conformal mapping of rectangles
    - Wave propagation in nonlinear media
    - Josephson junction dynamics

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

       \frac{\partial F}{\partial \phi} = \frac{1}{\sqrt{1 - m \sin^2\phi}}

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
        The incomplete elliptic integral of the first kind F(phi, m) evaluated
        at each pair of elements. Output dtype matches the promoted input dtype.

    Examples
    --------
    Basic usage:

    >>> phi = torch.tensor([0.0, 0.5, 1.0, 1.5707963267948966])  # 0, 0.5, 1.0, pi/2
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_f(phi, m)
    tensor([0.0000, 0.5155, 1.0854, 1.8541])

    The value at phi=0 is always 0:

    >>> phi = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_f(phi, m)
    tensor([0.])

    At phi=pi/2, this equals the complete elliptic integral K(m):

    >>> import math
    >>> phi = torch.tensor([math.pi / 2])
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_f(phi, m)
    tensor([1.8541])  # Same as complete K(0.5)

    Odd function property F(-phi, m) = -F(phi, m):

    >>> phi = torch.tensor([0.5])
    >>> m = torch.tensor([0.5])
    >>> f_pos = incomplete_legendre_elliptic_integral_f(phi, m)
    >>> f_neg = incomplete_legendre_elliptic_integral_f(-phi, m)
    >>> torch.allclose(f_neg, -f_pos)
    True

    Complex input:

    >>> phi = torch.tensor([1.0 + 0.1j])
    >>> m = torch.tensor([0.5])
    >>> incomplete_legendre_elliptic_integral_f(phi, m)
    tensor([1.0887+0.1050j])

    Autograd:

    >>> phi = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5], requires_grad=True)
    >>> y = incomplete_legendre_elliptic_integral_f(phi, m)
    >>> y.backward()
    >>> phi.grad
    tensor([1.1918])  # 1/sqrt(1 - 0.5 * sin^2(1))

    Notes
    -----
    - The implementation uses Carlson's symmetric form R_F for numerical
      stability and accuracy.
    - Broadcasting is applied to phi and m.

    See Also
    --------
    complete_legendre_elliptic_integral_k : Complete elliptic integral K(m)
    incomplete_legendre_elliptic_integral_e : Incomplete elliptic integral E(phi, m)
    carlson_elliptic_integral_r_f : Carlson's symmetric elliptic integral R_F
    """
    return torch.ops.torchscience.incomplete_legendre_elliptic_integral_f(
        phi, m
    )
