import torch
from torch import Tensor


def spherical_bessel_j_0(z: Tensor) -> Tensor:
    r"""
    Spherical Bessel function of the first kind of order zero.

    Computes the spherical Bessel function j_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The spherical Bessel function of the first kind of order zero is defined as:

    .. math::

       j_0(z) = \frac{\sin(z)}{z}

    This is related to the ordinary Bessel function by:

    .. math::

       j_0(z) = \sqrt{\frac{\pi}{2z}} J_{1/2}(z)

    Power series representation:

    .. math::

       j_0(z) = \sum_{k=0}^\infty \frac{(-1)^k z^{2k}}{(2k+1)!}
              = 1 - \frac{z^2}{6} + \frac{z^4}{120} - \cdots

    Special Values
    --------------
    - j_0(0) = 1 (removable singularity, limit is 1)
    - j_0(NaN) = NaN

    Domain
    ------
    - z: any real or complex value
    - j_0 is an entire function (no singularities or branch cuts)
    - The apparent singularity at z=0 is removable

    Algorithm
    ---------
    - For small |z|: Uses Taylor series j_0(z) = 1 - z^2/6 + z^4/120 - ...
    - For general z: Directly computes sin(z)/z
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The spherical Bessel function j_0 appears in many contexts:
    - Quantum mechanics: free particle wave functions in 3D
    - Scattering theory: partial wave expansion
    - Electrodynamics: multipole expansions
    - Acoustics: spherical wave propagation
    - Signal processing: spherical harmonic analysis

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

       \frac{d}{dz} j_0(z) = \frac{\cos(z) \cdot z - \sin(z)}{z^2} = -j_1(z)

    For small z, the Taylor series is used:

    .. math::

       \frac{d}{dz} j_0(z) = -\frac{z}{3} + \frac{z^3}{30} - \cdots

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} j_0(z) = -\frac{\sin(z)}{z} - \frac{2\cos(z)}{z^2}
                                + \frac{2\sin(z)}{z^3}

    At z=0, the limit gives j_0''(0) = -1/3.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The spherical Bessel function j_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> spherical_bessel_j_0(z)
    tensor([1.0000, 0.8415, 0.4546, 0.0470])

    Compare with sin(z)/z:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_j_0(z), torch.sin(z) / z)
    True

    Value at origin (removable singularity):

    >>> spherical_bessel_j_0(torch.tensor(0.0))
    tensor(1.)

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_j_0(z)
    tensor([0.8856+0.1174j, 0.4834+0.1509j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_j_0(z)
    >>> y.backward()
    >>> z.grad  # equals -j_1(2.0)
    tensor([-0.4353])

    Zeros of j_0 (same as zeros of sin(z)):

    >>> import math
    >>> z = torch.tensor([math.pi, 2*math.pi, 3*math.pi])
    >>> spherical_bessel_j_0(z).abs() < 1e-6
    tensor([True, True, True])

    Notes
    -----
    - The spherical Bessel functions arise naturally when solving the
      Helmholtz equation in spherical coordinates using separation of variables.
    - j_0(z) = sin(z)/z is the simplest spherical Bessel function and
      corresponds to the s-wave (l=0) in quantum mechanical scattering.

    See Also
    --------
    bessel_j_0 : Bessel function of the first kind of order zero
    """
    return torch.ops.torchscience.spherical_bessel_j_0(z)
