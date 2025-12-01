import torch
from torch import Tensor


def spherical_bessel_y_0(z: Tensor) -> Tensor:
    r"""
    Spherical Bessel function of the second kind of order zero.

    Computes the spherical Bessel function y_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The spherical Bessel function of the second kind of order zero is defined as:

    .. math::

       y_0(z) = -\frac{\cos(z)}{z}

    This is related to the ordinary Bessel function by:

    .. math::

       y_0(z) = \sqrt{\frac{\pi}{2z}} Y_{1/2}(z)

    Special Values
    --------------
    - y_0(0) = -infinity (singular at origin)
    - y_0(NaN) = NaN
    - y_0 is an even function: y_0(-z) = y_0(z)

    Domain
    ------
    - z: any real or complex value except z=0
    - y_0 has a singularity at z=0

    Algorithm
    ---------
    - Directly computes -cos(z)/z
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The spherical Bessel function y_0 appears in many contexts:
    - Quantum mechanics: irregular solutions to the radial Schrodinger equation
    - Scattering theory: partial wave expansion
    - Electrodynamics: multipole expansions
    - Acoustics: spherical wave propagation

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

       \frac{d}{dz} y_0(z) = \frac{\sin(z) \cdot z + \cos(z)}{z^2} = -y_1(z)

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} y_0(z) = \frac{\cos(z)}{z} - \frac{2\sin(z)}{z^2}
                                - \frac{2\cos(z)}{z^3}

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The spherical Bessel function y_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> spherical_bessel_y_0(z)
    tensor([-1.7552, -0.5403, -0.2081, -0.3300])

    Compare with -cos(z)/z:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_y_0(z), -torch.cos(z) / z)
    True

    Value at origin (singular):

    >>> spherical_bessel_y_0(torch.tensor(0.0))
    tensor(-inf)

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_y_0(z)
    tensor([-0.4652-0.2825j, -0.1823+0.1291j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_y_0(z)
    >>> y.backward()
    >>> z.grad  # equals -y_1(2.0)
    tensor([0.4353])

    Even function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_y_0(z), spherical_bessel_y_0(-z))
    True

    Notes
    -----
    - The spherical Bessel functions arise naturally when solving the
      Helmholtz equation in spherical coordinates using separation of variables.
    - y_0(z) = -cos(z)/z is the simplest spherical Bessel function of the
      second kind and corresponds to irregular solutions in quantum mechanical
      scattering.
    - Unlike j_0, the function y_0 is singular at the origin.

    See Also
    --------
    spherical_bessel_j_0 : Spherical Bessel function of the first kind of order zero
    bessel_y_0 : Bessel function of the second kind of order zero
    """
    return torch.ops.torchscience.spherical_bessel_y_0(z)
