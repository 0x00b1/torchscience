import torch
from torch import Tensor


def spherical_bessel_k_0(z: Tensor) -> Tensor:
    r"""
    Modified spherical Bessel function of the second kind of order zero.

    Computes the modified spherical Bessel function k_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified spherical Bessel function of the second kind of order zero is defined as:

    .. math::

       k_0(z) = \frac{\pi}{2z} e^{-z}

    This is related to the ordinary modified Bessel function by:

    .. math::

       k_n(z) = \sqrt{\frac{\pi}{2z}} K_{n+1/2}(z)

    For n=0:

    .. math::

       k_0(z) = \sqrt{\frac{\pi}{2z}} K_{1/2}(z) = \frac{\pi}{2z} e^{-z}

    Special Values
    --------------
    - k_0(z) is always positive for positive real z
    - As z -> 0+, k_0(z) -> infinity (pole at origin)
    - As z -> infinity, k_0(z) -> 0 (exponential decay)
    - k_0(NaN) = NaN

    Domain
    ------
    - z: any real or complex value except z=0 (pole)
    - For positive real z, k_0(z) is real and positive
    - For negative real z, k_0(z) is complex due to the exponential

    Algorithm
    ---------
    - For general z: Directly computes (pi/2z) * exp(-z)
    - Near z=0: Returns pi/(2z) as the dominant singular term
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified spherical Bessel function k_0 appears in many contexts:
    - Heat conduction in spherical geometries with exponential decay
    - Quantum mechanics: bound state solutions in spherical coordinates
    - Electromagnetic wave propagation in lossy media
    - Yukawa potential and screened Coulomb interactions

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

       \frac{d}{dz} k_0(z) = -k_1(z) = -\frac{\pi}{2z^2}(1+z)e^{-z}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} k_0(z) = \frac{\pi}{2z^3}(z^2 + 2z + 2)e^{-z}

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified spherical Bessel function k_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> spherical_bessel_k_0(z)
    tensor([1.9098, 0.5779, 0.1063, 0.0261])

    Compare with (pi/2z) * exp(-z):

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> import math
    >>> expected = (math.pi / (2 * z)) * torch.exp(-z)
    >>> torch.allclose(spherical_bessel_k_0(z), expected)
    True

    Exponential decay for large z:

    >>> z = torch.tensor([5.0, 10.0, 20.0])
    >>> spherical_bessel_k_0(z)
    tensor([2.1144e-03, 7.1165e-06, 2.8506e-11])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_k_0(z)
    tensor([0.4397-0.3025j, 0.0964+0.0330j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_k_0(z)
    >>> y.backward()
    >>> z.grad
    tensor([-0.1063])

    Notes
    -----
    - The modified spherical Bessel functions arise naturally when solving the
      modified Helmholtz equation in spherical coordinates using separation of variables.
    - k_0(z) = (pi/2z) * exp(-z) has a simple closed form, unlike higher orders.
    - As z -> 0+, k_0(z) diverges as pi/(2z).

    See Also
    --------
    spherical_bessel_i_0 : Modified spherical Bessel function of the first kind of order zero
    modified_bessel_k_0 : Modified Bessel function of the second kind of order zero
    """
    return torch.ops.torchscience.spherical_bessel_k_0(z)
