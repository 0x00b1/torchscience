import torch
from torch import Tensor


def spherical_bessel_j_1(z: Tensor) -> Tensor:
    r"""
    Spherical Bessel function of the first kind of order one.

    Computes the spherical Bessel function j_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The spherical Bessel function of the first kind of order one is defined as:

    .. math::

       j_1(z) = \frac{\sin(z)}{z^2} - \frac{\cos(z)}{z}

    This is related to the ordinary Bessel function by:

    .. math::

       j_1(z) = \sqrt{\frac{\pi}{2z}} J_{3/2}(z)

    Power series representation:

    .. math::

       j_1(z) = \sum_{k=0}^\infty \frac{(-1)^k z^{2k+1}}{(2k+3)!!(2k)!!}
              = \frac{z}{3} - \frac{z^3}{30} + \frac{z^5}{840} - \cdots

    Special Values
    --------------
    - j_1(0) = 0
    - j_1(NaN) = NaN
    - j_1 is an odd function: j_1(-z) = -j_1(z)

    Domain
    ------
    - z: any real or complex value
    - j_1 is an entire function (no singularities or branch cuts)
    - The apparent singularity at z=0 is removable

    Algorithm
    ---------
    - For small |z|: Uses Taylor series j_1(z) = z/3 - z^3/30 + z^5/840 - ...
    - For general z: Directly computes sin(z)/z^2 - cos(z)/z
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The spherical Bessel function j_1 appears in many contexts:
    - Quantum mechanics: p-wave (l=1) partial wave expansion
    - Scattering theory: dipole radiation patterns
    - Electrodynamics: multipole expansions (dipole term)
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
    The gradient is computed using the recurrence relation:

    .. math::

       \frac{d}{dz} j_1(z) = j_0(z) - \frac{2 j_1(z)}{z}

    For small z, the Taylor series is used:

    .. math::

       \frac{d}{dz} j_1(z) = \frac{1}{3} - \frac{z^2}{10} + \frac{z^4}{168} - \cdots

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} j_1(z) = -j_1(z) - \frac{2 j_0(z)}{z} + \frac{6 j_1(z)}{z^2}

    At z=0, using L'Hopital's rule or Taylor series: j_1''(0) = 0.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The spherical Bessel function j_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> spherical_bessel_j_1(z)
    tensor([0.0000, 0.3012, 0.4353, 0.3457])

    Compare with closed form sin(z)/z^2 - cos(z)/z:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_j_1(z), torch.sin(z) / z**2 - torch.cos(z) / z)
    True

    Value at origin:

    >>> spherical_bessel_j_1(torch.tensor(0.0))
    tensor(0.)

    Odd function symmetry:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_j_1(-z), -spherical_bessel_j_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_j_1(z)
    tensor([0.3366+0.1423j, 0.4631+0.0913j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_j_1(z)
    >>> y.backward()
    >>> z.grad  # equals j_0(2) - 2*j_1(2)/2
    tensor([0.0191])

    Notes
    -----
    - The spherical Bessel functions arise naturally when solving the
      Helmholtz equation in spherical coordinates using separation of variables.
    - j_1(z) corresponds to the p-wave (l=1) in quantum mechanical scattering,
      representing dipole contributions.
    - The recurrence relation j_{n+1}(z) = (2n+1)/z * j_n(z) - j_{n-1}(z) can be
      used to compute higher-order spherical Bessel functions from j_0 and j_1.

    See Also
    --------
    spherical_bessel_j_0 : Spherical Bessel function of the first kind of order zero
    bessel_j_1 : Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.spherical_bessel_j_1(z)
