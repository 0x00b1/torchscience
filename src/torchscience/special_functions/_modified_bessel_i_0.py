import torch
from torch import Tensor


def modified_bessel_i_0(z: Tensor) -> Tensor:
    r"""
    Modified Bessel function of the first kind of order zero.

    Computes the modified Bessel function I_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified Bessel function of the first kind of order zero is defined as:

    .. math::

       I_0(z) = \frac{1}{\pi} \int_0^\pi e^{z \cos(\theta)} \, d\theta

    Or equivalently via the power series:

    .. math::

       I_0(z) = \sum_{k=0}^\infty \frac{(z/2)^{2k}}{(k!)^2}

    Special Values
    --------------
    - I_0(0) = 1
    - I_0(+inf) = +inf
    - I_0(-inf) = +inf (even function)
    - I_0(NaN) = NaN

    Symmetry
    --------
    I_0 is an even function: I_0(-z) = I_0(z)

    Domain
    ------
    - z: any real or complex value
    - I_0 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses Chebyshev polynomial approximations (Cephes coefficients)
    - For |z| <= 8: Chebyshev expansion scaled by exp(|z|)
    - For |z| > 8: Asymptotic expansion exp(z) / sqrt(2*pi*z) * P(1/z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified Bessel function I_0 appears in many contexts:
    - Signal processing: Kaiser window functions for filter design
    - Statistics: von Mises distribution (circular normal distribution)
    - Statistics: Rice distribution (signal plus noise)
    - Physics: heat conduction in cylindrical coordinates
    - Physics: electromagnetic wave propagation
    - Communications: fading channel models

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using I_1:

    .. math::

       \frac{d}{dz} I_0(z) = I_1(z)

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} I_0(z) = I_0(z) - \frac{I_1(z)}{z}

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified Bessel function I_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> modified_bessel_i_0(z)
    tensor([1.0000, 1.2661, 2.2796, 4.8808])

    Even function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(modified_bessel_i_0(z), modified_bessel_i_0(-z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> modified_bessel_i_0(z)
    tensor([1.1563+0.3247j, 2.1516-0.5765j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = modified_bessel_i_0(z)
    >>> y.backward()
    >>> z.grad  # equals I_1(2.0)
    tensor([1.5906])

    .. warning:: Overflow for large arguments

       I_0(z) grows exponentially: I_0(z) ~ exp(z) / sqrt(2*pi*z) for large z.
       For reference:

       - I_0(100) ~ 1.1e42
       - I_0(700) ~ 1.5e303 (near float64 overflow)
       - I_0(710) overflows float64

       For large arguments, consider using the scaled form exp(-z) * I_0(z).

    Notes
    -----
    - Complex accuracy: The Chebyshev approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.
    - The implementation uses the Cephes library coefficients (public domain).

    See Also
    --------
    modified_bessel_i_1 : Modified Bessel function of order one
    """
    return torch.ops.torchscience.modified_bessel_i_0(z)
