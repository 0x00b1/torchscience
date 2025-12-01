import torch
from torch import Tensor


def complete_legendre_elliptic_integral_e(m: Tensor) -> Tensor:
    r"""
    Complete elliptic integral of the second kind E(m).

    Computes the complete elliptic integral of the second kind evaluated at
    each element of the input tensor.

    Mathematical Definition
    -----------------------
    The complete elliptic integral of the second kind is defined as:

    .. math::

       E(m) = \int_0^{\pi/2} \sqrt{1 - m \sin^2\theta} \, d\theta

    This is related to Carlson's symmetric elliptic integral R_G by:

    .. math::

       E(m) = 2 \cdot R_G(0, 1-m, 1)

    Special Values
    --------------
    - E(0) = pi/2
    - E(1) = 1

    Domain
    ------
    - For real m: The integral converges for m <= 1.
      At m = 1, E(1) = 1.
      For m > 1, the result is complex.
    - For complex m: Defined on the entire complex plane.

    Applications
    ------------
    The complete elliptic integral of the second kind appears in:
    - Arc length of ellipses: The circumference of an ellipse with
      semi-major axis a and semi-minor axis b is 4*a*E(1-b^2/a^2)
    - Period of a simple pendulum (large amplitude)
    - Electromagnetic calculations
    - Geodesics on an ellipsoid
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

    Autograd Support
    ----------------
    Gradients are fully supported when m.requires_grad is True.
    The gradient is:

    .. math::

       \frac{dE}{dm} = \frac{E(m) - K(m)}{2m}

    where K(m) is the complete elliptic integral of the first kind.

    Second-order derivatives are also supported.

    Parameters
    ----------
    m : Tensor
        Input tensor (the parameter). Can be floating-point or complex.

    Returns
    -------
    Tensor
        The complete elliptic integral of the second kind E(m) evaluated
        at each element of m. Output dtype matches input dtype.

    Examples
    --------
    Basic usage:

    >>> m = torch.tensor([0.0, 0.5, 0.9, 1.0])
    >>> complete_legendre_elliptic_integral_e(m)
    tensor([1.5708, 1.3506, 1.1048, 1.0000])

    The value at m=0 is pi/2:

    >>> import math
    >>> m = torch.tensor([0.0])
    >>> complete_legendre_elliptic_integral_e(m)
    tensor([1.5708])  # pi/2 approx 1.5708

    Complex input:

    >>> m = torch.tensor([0.5 + 0.1j])
    >>> complete_legendre_elliptic_integral_e(m)
    tensor([1.3450-0.0378j])

    Autograd:

    >>> m = torch.tensor([0.5], requires_grad=True)
    >>> y = complete_legendre_elliptic_integral_e(m)
    >>> y.backward()
    >>> m.grad
    tensor([-0.2192])

    Relation to ellipse arc length:

    >>> # Circumference of an ellipse with a=2, b=1
    >>> a, b = 2.0, 1.0
    >>> m = torch.tensor([1 - (b/a)**2])
    >>> circumference = 4 * a * complete_legendre_elliptic_integral_e(m)
    >>> circumference
    tensor([9.6884])

    Notes
    -----
    - The implementation uses Carlson's symmetric form R_G for numerical
      stability and accuracy.
    - For m near 1, special care is taken to handle the limiting behavior.

    See Also
    --------
    carlson_elliptic_integral_r_g : Carlson's symmetric elliptic integral R_G
    """
    return torch.ops.torchscience.complete_legendre_elliptic_integral_e(m)
