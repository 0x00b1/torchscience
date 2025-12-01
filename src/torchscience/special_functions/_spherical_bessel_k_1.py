import torch
from torch import Tensor


def spherical_bessel_k_1(z: Tensor) -> Tensor:
    r"""
    Modified spherical Bessel function of the second kind of order one.

    Computes the modified spherical Bessel function k_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified spherical Bessel function of the second kind of order one is defined as:

    .. math::

       k_1(z) = \frac{\pi}{2z^2}(1+z) e^{-z}

    This is related to the ordinary modified Bessel function by:

    .. math::

       k_1(z) = \sqrt{\frac{\pi}{2z}} K_{3/2}(z)

    where K is the modified Bessel function of the second kind.

    Special Values
    --------------
    - k_1(0) = infinity (pole at origin)
    - k_1(NaN) = NaN
    - k_1 is an even function: k_1(-z) = k_1(z) for real z in proper branch
    - As z -> infinity, k_1(z) -> 0 (exponential decay)

    Domain
    ------
    - z: any real or complex value except z=0
    - k_1 has a pole at z=0 (second-order pole)
    - For negative real z, the function is defined via analytic continuation

    Algorithm
    ---------
    - For |z| near 0: Returns infinity (pole behavior)
    - For general z: Directly computes (pi/2z^2)(1+z)e^(-z)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified spherical Bessel function k_1 appears in many contexts:
    - Quantum mechanics: radial wave functions with exponential decay
    - Heat conduction: temperature distributions in spherical geometries
    - Electrodynamics: near-field expansions and multipole moments
    - Diffusion problems: solutions in spherical coordinates
    - Gravitational physics: Yukawa-type potentials

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using the formula:

    .. math::

       \frac{d}{dz} k_1(z) = -\frac{\pi}{2z^3}(2 + 2z + z^2) e^{-z}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} k_1(z) = \frac{\pi}{2z^4}(6 + 6z + 3z^2 + z^3) e^{-z}

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified spherical Bessel function k_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> spherical_bessel_k_1(z)
    tensor([5.7620, 1.4282, 0.2392, 0.0457])

    Compare with closed form (pi/2z^2)(1+z)e^(-z):

    >>> import math
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(spherical_bessel_k_1(z),
    ...                (math.pi / (2 * z**2)) * (1 + z) * torch.exp(-z))
    True

    Pole at origin:

    >>> spherical_bessel_k_1(torch.tensor(0.0))
    tensor(inf)

    Exponential decay at large z:

    >>> z = torch.tensor([10.0, 20.0, 50.0])
    >>> spherical_bessel_k_1(z)  # Rapidly approaches 0
    tensor([7.9577e-06, 4.7756e-10, 3.2305e-23])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> spherical_bessel_k_1(z)
    tensor([0.9093-0.8610j, 0.2142+0.1070j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = spherical_bessel_k_1(z)
    >>> y.backward()
    >>> z.grad  # d/dz k_1(z)
    tensor([-0.1757])

    Notes
    -----
    - The modified spherical Bessel functions arise naturally when solving the
      modified Helmholtz equation in spherical coordinates using separation of variables.
    - k_1(z) represents decaying solutions, in contrast to i_1(z) which grows.
    - The recurrence relation k_{n+1}(z) = k_{n-1}(z) + (2n+1)/z * k_n(z) can be
      used to compute higher-order modified spherical Bessel functions from k_0 and k_1.
    - Unlike i_n which is entire, k_n has a pole at z=0.

    See Also
    --------
    spherical_bessel_k_0 : Modified spherical Bessel function of the second kind of order zero
    spherical_bessel_i_1 : Modified spherical Bessel function of the first kind of order one
    modified_bessel_k_1 : Modified Bessel function of the second kind of order one
    """
    return torch.ops.torchscience.spherical_bessel_k_1(z)
