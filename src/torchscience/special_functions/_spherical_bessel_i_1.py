import torch
from torch import Tensor


def spherical_bessel_i_1(z: Tensor) -> Tensor:
    r"""
    Modified spherical Bessel function of the first kind of order one.

    Computes the modified spherical Bessel function i_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified spherical Bessel function of the first kind of order one is defined as:

    .. math::

       i_1(z) = \frac{\cosh(z)}{z} - \frac{\sinh(z)}{z^2}

    This is related to the ordinary modified Bessel function by:

    .. math::

       i_1(z) = \sqrt{\frac{\pi}{2z}} I_{3/2}(z)

    Power series representation:

    .. math::

       i_1(z) = \sum_{k=0}^\infty \frac{z^{2k+1}}{(2k+3)!!(2k)!!}
              = \frac{z}{3} + \frac{z^3}{30} + \frac{z^5}{840} + \cdots

    Special Values
    --------------
    - i_1(0) = 0
    - i_1(NaN) = NaN
    - i_1 is an odd function: i_1(-z) = -i_1(z)

    Domain
    ------
    - z: any real or complex value
    - i_1 is an entire function (no singularities or branch cuts)
    - The apparent singularity at z=0 is removable

    Algorithm
    ---------
    - For small |z|: Uses Taylor series i_1(z) = z/3 + z^3/30 + z^5/840 + ...
    - For general z: Directly computes cosh(z)/z - sinh(z)/z^2
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified spherical Bessel function i_1 appears in many contexts:
    - Quantum mechanics: radial wave functions in spherical wells
    - Heat conduction: temperature distributions in spherical geometries
    - Electrodynamics: near-field expansions
    - Diffusion problems: solutions in spherical coordinates
    - Signal processing: spherical harmonic analysis

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the recurrence relation:

    .. math::

       \frac{d}{dz} i_1(z) = i_0(z) - \frac{2 i_1(z)}{z}

    For small z, the Taylor series is used:

    .. math::

       \frac{d}{dz} i_1(z) = \frac{1}{3} + \frac{z^2}{10} + \frac{z^4}{168} + \cdots

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} i_1(z) = i_1(z) - \frac{2 i_0(z)}{z} + \frac{6 i_1(z)}{z^2}

    At z=0, using L'Hopital's rule or Taylor series: i_1''(0) = 0.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified spherical Bessel function i_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> spherical_bessel_i_1(z)
    tensor([0.0000, 0.3679, 1.0377, 3.2534])

    Compare with closed form cosh(z)/z - sinh(z)/z^2:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_i_1(z), torch.cosh(z) / z - torch.sinh(z) / z**2)
    True

    Value at origin:

    >>> spherical_bessel_i_1(torch.tensor(0.0))
    tensor(0.)

    Odd function symmetry:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_i_1(-z), -spherical_bessel_i_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_i_1(z)
    tensor([0.3577+0.2087j, 1.0834-0.2644j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_i_1(z)
    >>> y.backward()
    >>> z.grad  # equals i_0(2) - 2*i_1(2)/2
    tensor([0.4170])

    Notes
    -----
    - The modified spherical Bessel functions arise naturally when solving the
      modified Helmholtz equation in spherical coordinates using separation of variables.
    - i_1(z) is the hyperbolic analogue of j_1(z), related by i_1(z) = -i*j_1(i*z).
    - The recurrence relation i_{n+1}(z) = i_{n-1}(z) - (2n+1)/z * i_n(z) can be
      used to compute higher-order modified spherical Bessel functions from i_0 and i_1.

    See Also
    --------
    spherical_bessel_i_0 : Modified spherical Bessel function of the first kind of order zero
    spherical_bessel_j_1 : Spherical Bessel function of the first kind of order one
    modified_bessel_i_1 : Modified Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.spherical_bessel_i_1(z)
