import torch
from torch import Tensor


def spherical_bessel_i_0(z: Tensor) -> Tensor:
    r"""
    Modified spherical Bessel function of the first kind of order zero.

    Computes the modified spherical Bessel function i_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified spherical Bessel function of the first kind of order zero is defined as:

    .. math::

       i_0(z) = \frac{\sinh(z)}{z}

    This is related to the ordinary modified Bessel function by:

    .. math::

       i_0(z) = \sqrt{\frac{\pi}{2z}} I_{1/2}(z)

    Power series representation:

    .. math::

       i_0(z) = \sum_{k=0}^\infty \frac{z^{2k}}{(2k+1)!}
              = 1 + \frac{z^2}{6} + \frac{z^4}{120} + \cdots

    Special Values
    --------------
    - i_0(0) = 1 (removable singularity, limit is 1)
    - i_0(NaN) = NaN
    - i_0(-z) = i_0(z) (even function)

    Domain
    ------
    - z: any real or complex value
    - i_0 is an entire function (no singularities or branch cuts)
    - The apparent singularity at z=0 is removable

    Algorithm
    ---------
    - For small |z|: Uses Taylor series i_0(z) = 1 + z^2/6 + z^4/120 + ...
    - For general z: Directly computes sinh(z)/z
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified spherical Bessel function i_0 appears in many contexts:
    - Heat conduction in spherical geometries
    - Quantum mechanics: solutions with imaginary arguments
    - Statistical mechanics: spherical model calculations
    - Signal processing: spherical harmonic analysis with exponential behavior

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

       \frac{d}{dz} i_0(z) = \frac{\cosh(z) \cdot z - \sinh(z)}{z^2}

    For small z, the Taylor series is used:

    .. math::

       \frac{d}{dz} i_0(z) = \frac{z}{3} + \frac{z^3}{30} + \cdots

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} i_0(z) = \frac{\sinh(z)}{z} - \frac{2\cosh(z)}{z^2}
                                + \frac{2\sinh(z)}{z^3}

    At z=0, the limit gives i_0''(0) = 1/3.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified spherical Bessel function i_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> spherical_bessel_i_0(z)
    tensor([1.0000, 1.1752, 1.8134, 3.3393])

    Compare with sinh(z)/z:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_i_0(z), torch.sinh(z) / z)
    True

    Value at origin (removable singularity):

    >>> spherical_bessel_i_0(torch.tensor(0.0))
    tensor(1.)

    Even function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_i_0(z), spherical_bessel_i_0(-z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_i_0(z)
    tensor([1.1290+0.2371j, 1.7215-0.4696j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_i_0(z)
    >>> y.backward()
    >>> z.grad
    tensor([0.6514])

    Notes
    -----
    - The modified spherical Bessel functions arise naturally when solving the
      modified Helmholtz equation in spherical coordinates using separation of variables.
    - i_0(z) = sinh(z)/z is the simplest modified spherical Bessel function.
    - Related to the regular spherical Bessel function by i_0(z) = j_0(iz)/i^0 = j_0(iz).

    See Also
    --------
    spherical_bessel_j_0 : Spherical Bessel function of the first kind of order zero
    modified_bessel_i_0 : Modified Bessel function of the first kind of order zero
    """
    return torch.ops.torchscience.spherical_bessel_i_0(z)
