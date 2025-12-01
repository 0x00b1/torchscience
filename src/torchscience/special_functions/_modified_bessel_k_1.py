import torch
from torch import Tensor


def modified_bessel_k_1(z: Tensor) -> Tensor:
    r"""
    Modified Bessel function of the second kind of order one.

    Computes the modified Bessel function K_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified Bessel function of the second kind of order one is defined as:

    .. math::

       K_1(z) = \int_0^\infty e^{-z \cosh t} \cosh t \, dt

    It can also be expressed using the integral representation:

    .. math::

       K_1(z) = \frac{z}{2} \int_0^\infty \frac{e^{-t}}{t^2} e^{-z^2/(4t)} \, dt

    Special Values
    --------------
    - K_1(0) = +infinity (singularity: K_1(z) ~ 1/z as z -> 0+)
    - K_1(+inf) = 0 (exponential decay)
    - K_1(NaN) = NaN
    - K_1(z < 0) = NaN (for real z; not defined on negative real axis)

    Domain
    ------
    - z > 0 for real arguments (K_1 is only defined for positive real arguments)
    - For complex z, the principal branch is defined with cut along
      the negative real axis
    - The function has a pole-like singularity at z = 0: K_1(z) ~ 1/z

    Algorithm
    ---------
    - Uses Chebyshev polynomial expansions (Cephes coefficients)
    - For |z| <= 2: K_1(z) = ln(z/2)*I_1(z) + (1/z)*P(z^2 - 2)
    - For |z| > 2: K_1(z) = exp(-z)/sqrt(z) * Q(8/z - 2)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified Bessel function K_1 appears in many contexts:
    - Physics: Yukawa potential gradients, nuclear physics
    - Electromagnetics: dipole fields with screening
    - Quantum mechanics: radial wavefunctions
    - Statistics: derivatives of K_0-based distributions

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

       \frac{d}{dz} K_1(z) = -K_0(z) - \frac{K_1(z)}{z}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} K_1(z) = K_1(z) + \frac{K_0(z)}{z} + \frac{2 K_1(z)}{z^2}

    Parameters
    ----------
    z : Tensor
        Input tensor. Must be positive for real tensors.
        Can be complex (with branch cut along negative real axis).

    Returns
    -------
    Tensor
        The modified Bessel function K_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> modified_bessel_k_1(z)
    tensor([1.6564, 0.6019, 0.1399, 0.0402])

    Negative arguments return NaN:

    >>> z = torch.tensor([-1.0, -2.0])
    >>> modified_bessel_k_1(z).isnan()
    tensor([True, True])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 + 0.5j])
    >>> modified_bessel_k_1(z)
    tensor([0.4605-0.4171j, 0.1103-0.0915j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = modified_bessel_k_1(z)
    >>> y.backward()
    >>> z.grad  # equals -K_0(2.0) - K_1(2.0)/2.0
    tensor([-0.1838])

    K_1(z) > 0 for all z > 0:

    >>> z = torch.tensor([0.1, 1.0, 10.0])
    >>> (modified_bessel_k_1(z) > 0).all()
    tensor(True)

    Near-zero behavior (K_1(z) ~ 1/z):

    >>> z = torch.tensor([0.01])
    >>> modified_bessel_k_1(z) * z  # approximately 1
    tensor([1.0050])

    .. warning:: Singularity at z = 0

       K_1(z) has a pole-like singularity at z = 0. As z -> 0+,
       K_1(z) ~ 1/z. Computations near z = 0 may have
       reduced precision or overflow.

    Notes
    -----
    - Complex accuracy: The Cephes approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.
    - The implementation uses the Cephes library coefficients (public domain).
    - For z on the negative real axis, the function returns NaN for real
      inputs. For complex inputs, the principal branch is used.

    See Also
    --------
    modified_bessel_k_0 : Modified Bessel function of the second kind of order zero
    modified_bessel_i_0 : Modified Bessel function of the first kind of order zero
    modified_bessel_i_1 : Modified Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.modified_bessel_k_1(z)
