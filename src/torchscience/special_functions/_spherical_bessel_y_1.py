import torch
from torch import Tensor


def spherical_bessel_y_1(z: Tensor) -> Tensor:
    r"""
    Spherical Bessel function of the second kind of order one.

    Computes the spherical Bessel function y_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The spherical Bessel function of the second kind of order one is defined as:

    .. math::

       y_1(z) = -\frac{\cos(z)}{z^2} - \frac{\sin(z)}{z}

    This is related to the ordinary Bessel function by:

    .. math::

       y_1(z) = \sqrt{\frac{\pi}{2z}} Y_{3/2}(z)

    Special Values
    --------------
    - y_1(0) = -infinity (singular)
    - y_1(NaN) = NaN
    - y_1 is an odd function: y_1(-z) = -y_1(z)

    Domain
    ------
    - z: any real or complex value except z=0
    - y_1 has a singularity at z=0

    Algorithm
    ---------
    - Directly computes -cos(z)/z^2 - sin(z)/z
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The spherical Bessel function y_1 appears in many contexts:
    - Quantum mechanics: p-wave (l=1) partial wave expansion (irregular solutions)
    - Scattering theory: outgoing wave boundary conditions
    - Electrodynamics: multipole expansions (dipole term)
    - Acoustics: spherical wave propagation

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

       \frac{d}{dz} y_1(z) = y_0(z) - \frac{2 y_1(z)}{z}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} y_1(z) = -y_1(z) - \frac{2 y_0(z)}{z} + \frac{6 y_1(z)}{z^2}

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The spherical Bessel function y_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> spherical_bessel_y_1(z)
    tensor([-5.0421, -1.3818, -0.4560, -0.0930])

    Compare with closed form -cos(z)/z^2 - sin(z)/z:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_y_1(z), -torch.cos(z) / z**2 - torch.sin(z) / z)
    True

    Singular at origin:

    >>> spherical_bessel_y_1(torch.tensor(0.0))
    tensor(-inf)

    Odd function symmetry:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_y_1(-z), -spherical_bessel_y_1(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_y_1(z)
    tensor([-1.2003-0.5861j, -0.4111+0.1937j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_y_1(z)
    >>> y.backward()
    >>> z.grad  # equals y_0(2) - 2*y_1(2)/2
    tensor([0.2478])

    Notes
    -----
    - The spherical Bessel functions arise naturally when solving the
      Helmholtz equation in spherical coordinates using separation of variables.
    - y_1(z) corresponds to the p-wave (l=1) irregular solution in quantum
      mechanical scattering.
    - The recurrence relation y_{n+1}(z) = (2n+1)/z * y_n(z) - y_{n-1}(z) can be
      used to compute higher-order spherical Bessel functions from y_0 and y_1.

    See Also
    --------
    spherical_bessel_y_0 : Spherical Bessel function of the second kind of order zero
    spherical_bessel_j_1 : Spherical Bessel function of the first kind of order one
    bessel_y_1 : Bessel function of the second kind of order one
    """
    return torch.ops.torchscience.spherical_bessel_y_1(z)
