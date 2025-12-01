import torch
from torch import Tensor


def bessel_j_1(z: Tensor) -> Tensor:
    r"""
    Bessel function of the first kind of order one.

    Computes the Bessel function J_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The Bessel function of the first kind of order one is defined as:

    .. math::

       J_1(z) = \frac{1}{\pi} \int_0^\pi \cos(z \sin(\theta) - \theta) \, d\theta

    Or equivalently via the power series:

    .. math::

       J_1(z) = \sum_{k=0}^\infty \frac{(-1)^k (z/2)^{2k+1}}{k! (k+1)!}

    Special Values
    --------------
    - J_1(0) = 0
    - J_1(+inf) = 0
    - J_1(-inf) = 0
    - J_1(NaN) = NaN

    Symmetry
    --------
    J_1 is an odd function: J_1(-z) = -J_1(z)

    Domain
    ------
    - z: any real or complex value
    - J_1 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses rational polynomial approximations (Cephes coefficients)
    - For |z| <= 5: Rational polynomial approximation J_1(z) = z * R(z^2)
    - For |z| > 5: Asymptotic expansion J_1(z) ~ sqrt(2/(pi*z)) * [P*cos(theta) - Q*sin(theta)]
      where theta = z - 3*pi/4
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Bessel function J_1 appears in many contexts:
    - Signal processing: circular aperture diffraction (Airy pattern)
    - Physics: cylindrical waveguides and resonators
    - Electromagnetics: antenna radiation patterns
    - Acoustics: vibrations of circular membranes
    - Optics: diffraction through circular apertures

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

       \frac{d}{dz} J_1(z) = J_0(z) - \frac{J_1(z)}{z}

    At z=0, the limit gives J_1'(0) = 1/2.

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} J_1(z) = -J_1(z) - \frac{J_0(z)}{z} + \frac{2 J_1(z)}{z^2}

    At z=0, the limit gives J_1''(0) = 0.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Bessel function J_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> bessel_j_1(z)
    tensor([0.0000, 0.4401, 0.5767, 0.3391])

    Odd function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(bessel_j_1(-z), -bessel_j_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> bessel_j_1(z)
    tensor([0.4707+0.1754j, 0.5614-0.2081j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = bessel_j_1(z)
    >>> y.backward()
    >>> z.grad  # equals J_0(2.0) - J_1(2.0)/2.0
    tensor([-0.0649])

    Zeros of J_1:

    >>> # First few positive zeros: ~3.8317, ~7.0156, ~10.1735
    >>> z = torch.tensor([3.8317])
    >>> bessel_j_1(z).abs() < 1e-4
    tensor([True])

    .. warning:: Oscillatory behavior

       J_1(z) oscillates with decreasing amplitude for large z:
       J_1(z) ~ sqrt(2/(pi*z)) * cos(z - 3*pi/4) for large real z.
       The function has infinitely many real zeros.

    Notes
    -----
    - Complex accuracy: The Cephes approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.
    - The implementation uses the Cephes library coefficients (public domain).

    See Also
    --------
    bessel_j_0 : Bessel function of the first kind of order zero
    """
    return torch.ops.torchscience.bessel_j_1(z)
