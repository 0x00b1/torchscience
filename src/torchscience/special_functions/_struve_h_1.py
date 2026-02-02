import torch
from torch import Tensor


def struve_h_1(z: Tensor) -> Tensor:
    r"""
    Struve function of order one.

    Computes the Struve function H_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The Struve function of order one is defined as:

    .. math::

       \mathbf{H}_1(z) = \frac{2}{\pi} \sum_{k=0}^\infty \frac{(-1)^k (z/2)^{2k+2}}{\Gamma(k + 3/2) \Gamma(k + 5/2)}

    Or equivalently via the integral representation:

    .. math::

       \mathbf{H}_1(z) = \frac{2z}{\pi} \int_0^{\pi/2} \sin^2(\theta) \sin(z \cos(\theta)) \, d\theta

    Special Values
    --------------
    - H_1(0) = 0
    - H_1(+inf) = 2/pi (approaches from oscillation)
    - H_1(-inf) = 2/pi (approaches from oscillation)
    - H_1(NaN) = NaN

    Symmetry
    --------
    H_1 is an even function: H_1(-z) = H_1(z)

    Domain
    ------
    - z: any real or complex value
    - H_1 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses power series expansion for small |z|
    - Uses asymptotic expansion for large |z|:
      H_1(z) ~ Y_1(z) + (2/pi)[1 + (1/z^2) - (3/z^4) + ...]
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Struve function H_1 appears in many contexts:
    - Electromagnetics: radiation impedance of circular pistons
    - Acoustics: sound radiation from vibrating circular membranes
    - Fluid dynamics: viscous flow around cylinders
    - Optics: diffraction from circular apertures
    - Antenna theory: mutual impedance calculations

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dz} \mathbf{H}_1(z) = \mathbf{H}_0(z) - \frac{\mathbf{H}_1(z)}{z}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} \mathbf{H}_1(z) = \frac{2}{\pi} - \mathbf{H}_1(z) + \frac{\mathbf{H}_1(z)}{z^2} - \frac{\mathbf{H}_0(z)}{z}

    At z=0, the derivative H_1'(0) = 2/(3*pi) and H_1''(0) = 0.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Struve function H_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> struve_h_1(z)
    tensor([0.0000, 0.1981, 0.5296, 0.7747])

    Even function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(struve_h_1(-z), struve_h_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> struve_h_1(z)
    tensor([0.2102+0.1289j, 0.5432-0.2198j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = struve_h_1(z)
    >>> y.backward()
    >>> z.grad  # equals H_0(2.0) - H_1(2.0)/2.0
    tensor([0.5256])

    Asymptotic limit:

    >>> # H_1(z) -> 2/pi as z -> infinity
    >>> z = torch.tensor([50.0, 100.0])
    >>> struve_h_1(z)
    tensor([0.6366, 0.6366])  # approximately 2/pi = 0.6366...

    .. warning:: Oscillatory behavior

       For large z, H_1(z) oscillates around the asymptotic value 2/pi.
       The oscillations decay as 1/z for large real z.

    Notes
    -----
    - The Struve function is named after Hermann Struve (1854-1920).
    - H_1(z) satisfies the inhomogeneous Bessel equation:
      z^2 y'' + z y' + (z^2 - 1) y = 2z^2/(pi)
    - Complex accuracy: approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.

    See Also
    --------
    struve_h_0 : Struve function of order zero
    bessel_y_1 : Bessel function of the second kind of order one
    """
    return torch.ops.torchscience.struve_h_1(z)
