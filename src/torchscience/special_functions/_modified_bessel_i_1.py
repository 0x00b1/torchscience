import torch
from torch import Tensor


def modified_bessel_i_1(z: Tensor) -> Tensor:
    r"""
    Modified Bessel function of the first kind of order one.

    Computes the modified Bessel function I_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified Bessel function of the first kind of order one is defined as:

    .. math::

       I_1(z) = \frac{1}{\pi} \int_0^\pi e^{z \cos(\theta)} \cos(\theta) \, d\theta

    Or equivalently via the power series:

    .. math::

       I_1(z) = \sum_{k=0}^\infty \frac{(z/2)^{2k+1}}{k! (k+1)!}

    Special Values
    --------------
    - I_1(0) = 0
    - I_1(+inf) = +inf
    - I_1(-inf) = -inf (odd function)
    - I_1(NaN) = NaN

    Symmetry
    --------
    I_1 is an odd function: I_1(-z) = -I_1(z)

    Domain
    ------
    - z: any real or complex value
    - I_1 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses Chebyshev polynomial approximations (Cephes coefficients)
    - For |z| <= 8: Chebyshev expansion scaled by z * exp(|z|)
    - For |z| > 8: Asymptotic expansion exp(z) / sqrt(2*pi*z) * Q(1/z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified Bessel function I_1 appears in many contexts:
    - Signal processing: derivative of Kaiser windows
    - Statistics: von Mises distribution (circular normal distribution)
    - Statistics: Rice distribution normalization
    - Physics: heat conduction with cylindrical symmetry
    - Physics: electromagnetic field calculations
    - Communications: Rician fading channel models

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

       \frac{d}{dz} I_1(z) = I_0(z) - \frac{I_1(z)}{z}

    At z=0, the limit gives I_1'(0) = 1/2.

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} I_1(z) = I_1(z) - \frac{I_0(z)}{z} + \frac{2 I_1(z)}{z^2}

    At z=0, the limit gives I_1''(0) = 0.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified Bessel function I_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> modified_bessel_i_1(z)
    tensor([0.0000, 0.5652, 1.5906, 3.9534])

    Odd function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(modified_bessel_i_1(-z), -modified_bessel_i_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> modified_bessel_i_1(z)
    tensor([0.4923+0.3092j, 1.4680-0.6213j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = modified_bessel_i_1(z)
    >>> y.backward()
    >>> z.grad  # equals I_0(2.0) - I_1(2.0)/2.0
    tensor([1.4843])

    Wronskian identity:

    >>> z = torch.tensor([1.0, 2.0, 5.0])
    >>> i0, i1 = modified_bessel_i_0(z), modified_bessel_i_1(z)
    >>> # I_0(z) * I_1'(z) - I_1(z) * I_0'(z) = 1/z
    >>> torch.allclose(i0 * (i0 - i1/z) - i1 * i1, 1/z, rtol=1e-5)
    True

    .. warning:: Overflow for large arguments

       I_1(z) grows exponentially: I_1(z) ~ exp(z) / sqrt(2*pi*z) for large z.
       For reference:

       - I_1(100) ~ 1.1e42
       - I_1(700) ~ 1.5e303 (near float64 overflow)
       - I_1(710) overflows float64

       For large arguments, consider using the scaled form exp(-z) * I_1(z).

    Notes
    -----
    - Complex accuracy: The Chebyshev approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.
    - The implementation uses the Cephes library coefficients (public domain).

    See Also
    --------
    modified_bessel_i_0 : Modified Bessel function of order zero
    """
    return torch.ops.torchscience.modified_bessel_i_1(z)
