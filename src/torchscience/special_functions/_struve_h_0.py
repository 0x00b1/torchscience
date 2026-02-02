import torch
from torch import Tensor


def struve_h_0(z: Tensor) -> Tensor:
    r"""
    Struve function of order zero.

    Computes the Struve function H_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The Struve function of order zero is defined as:

    .. math::

       \mathbf{H}_0(z) = \frac{2}{\pi} \sum_{k=0}^\infty \frac{(-1)^k (z/2)^{2k+1}}{(\Gamma(k + 3/2))^2}

    Or equivalently via the integral representation:

    .. math::

       \mathbf{H}_0(z) = \frac{2}{\pi} \int_0^{\pi/2} \sin(z \cos(\theta)) \, d\theta

    Special Values
    --------------
    - H_0(0) = 0
    - H_0(+inf) oscillates, but H_0(z) - Y_0(z) -> 0 as z -> +inf
    - H_0(-inf) oscillates
    - H_0(NaN) = NaN

    Symmetry
    --------
    H_0 is an odd function: H_0(-z) = -H_0(z)

    Domain
    ------
    - z: any real or complex value
    - H_0 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses power series expansion for small |z|
    - Uses asymptotic expansion for large |z|:
      H_0(z) ~ Y_0(z) + (2/pi)[1 - (1/z^2) + (9/z^4) - ...]
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Struve function H_0 appears in many contexts:
    - Electromagnetics: radiation from antennas and apertures
    - Acoustics: sound radiation from circular pistons
    - Fluid dynamics: impulsive motion of cylinders in fluid
    - Optics: diffraction through circular apertures
    - Heat conduction: transient heat flow in cylindrical geometries

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

       \frac{d}{dz} \mathbf{H}_0(z) = \frac{2}{\pi} - \mathbf{H}_1(z)

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} \mathbf{H}_0(z) = \mathbf{H}_0(z) - \frac{\mathbf{H}_1(z)}{z} - \frac{2}{\pi z}

    At z=0, the limit gives H_0''(0) = 2/(3*pi).

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Struve function H_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> struve_h_0(z)
    tensor([0.0000, 0.5683, 0.7904, 0.5748])

    Odd function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(struve_h_0(-z), -struve_h_0(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> struve_h_0(z)
    tensor([0.5931+0.2371j, 0.7731-0.2618j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = struve_h_0(z)
    >>> y.backward()
    >>> z.grad  # equals 2/pi - H_1(2.0)
    tensor([0.0512])

    Relation to Bessel Y_0 for large z:

    >>> # H_0(z) - Y_0(z) -> 0 as z -> infinity
    >>> z = torch.tensor([50.0])
    >>> h0 = struve_h_0(z)
    >>> # The difference approaches zero for large z

    .. warning:: Oscillatory behavior

       For large z, H_0(z) oscillates similarly to Y_0(z). The asymptotic
       behavior is H_0(z) ~ Y_0(z) + O(1/z) for large real z.

    Notes
    -----
    - The Struve function is named after Hermann Struve (1854-1920).
    - H_0(z) satisfies the inhomogeneous Bessel equation:
      z^2 y'' + z y' + z^2 y = 2z/pi
    - Complex accuracy: approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.

    See Also
    --------
    struve_h_1 : Struve function of order one
    bessel_y_0 : Bessel function of the second kind of order zero
    """
    return torch.ops.torchscience.struve_h_0(z)
