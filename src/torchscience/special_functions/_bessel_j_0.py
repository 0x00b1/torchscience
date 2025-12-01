import torch
from torch import Tensor


def bessel_j_0(z: Tensor) -> Tensor:
    r"""
    Bessel function of the first kind of order zero.

    Computes the Bessel function J_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The Bessel function of the first kind of order zero is defined as:

    .. math::

       J_0(z) = \frac{1}{\pi} \int_0^\pi \cos(z \sin(\theta)) \, d\theta

    Or equivalently via the power series:

    .. math::

       J_0(z) = \sum_{k=0}^\infty \frac{(-1)^k (z/2)^{2k}}{(k!)^2}

    Special Values
    --------------
    - J_0(0) = 1
    - J_0(+inf) = 0
    - J_0(-inf) = 0
    - J_0(NaN) = NaN

    Symmetry
    --------
    J_0 is an even function: J_0(-z) = J_0(z)

    Domain
    ------
    - z: any real or complex value
    - J_0 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses rational polynomial approximations (Cephes coefficients)
    - For |z| <= 5: Rational polynomial approximation factored through zeros
    - For |z| > 5: Asymptotic expansion J_0(z) ~ sqrt(2/(pi*z)) * [P*cos(theta) - Q*sin(theta)]
      where theta = z - pi/4
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Bessel function J_0 appears in many contexts:
    - Signal processing: Fourier transform of circular functions
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

       \frac{d}{dz} J_0(z) = -J_1(z)

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} J_0(z) = \frac{J_1(z)}{z} - J_0(z)

    At z=0, the limit gives J_0''(0) = -1/2.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Bessel function J_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> bessel_j_0(z)
    tensor([1.0000, 0.7652, 0.2239, -0.2601])

    Even function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(bessel_j_0(-z), bessel_j_0(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> bessel_j_0(z)
    tensor([0.8102+0.1122j, 0.2645+0.1395j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = bessel_j_0(z)
    >>> y.backward()
    >>> z.grad  # equals -J_1(2.0)
    tensor([-0.5767])

    Zeros of J_0:

    >>> # First few positive zeros: ~2.4048, ~5.5201, ~8.6537
    >>> z = torch.tensor([2.4048])
    >>> bessel_j_0(z).abs() < 1e-4
    tensor([True])

    .. warning:: Oscillatory behavior

       J_0(z) oscillates with decreasing amplitude for large z:
       J_0(z) ~ sqrt(2/(pi*z)) * cos(z - pi/4) for large real z.
       The function has infinitely many real zeros.

    Notes
    -----
    - Complex accuracy: The Cephes approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.
    - The implementation uses the Cephes library coefficients (public domain).

    See Also
    --------
    bessel_j_1 : Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.bessel_j_0(z)
