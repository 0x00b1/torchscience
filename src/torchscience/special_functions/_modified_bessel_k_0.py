import torch
from torch import Tensor


def modified_bessel_k_0(z: Tensor) -> Tensor:
    r"""
    Modified Bessel function of the second kind of order zero.

    Computes the modified Bessel function K_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified Bessel function of the second kind of order zero is defined as:

    .. math::

       K_0(z) = \int_0^\infty e^{-z \cosh t} \, dt

    It can also be expressed in terms of the modified Bessel function of
    the first kind:

    .. math::

       K_0(z) = -\left[\ln\frac{z}{2} + \gamma\right] I_0(z)
                + \sum_{k=1}^\infty \frac{(z/2)^{2k}}{(k!)^2} H_k

    where H_k is the k-th harmonic number and gamma is the Euler-Mascheroni
    constant.

    Special Values
    --------------
    - K_0(0) = +infinity (logarithmic singularity)
    - K_0(+inf) = 0 (exponential decay)
    - K_0(NaN) = NaN
    - K_0(z < 0) = NaN (for real z; not defined on negative real axis)

    Domain
    ------
    - z > 0 for real arguments (K_0 is only defined for positive real arguments)
    - For complex z, the principal branch is defined with cut along
      the negative real axis
    - The function has a logarithmic singularity at z = 0

    Algorithm
    ---------
    - Uses Chebyshev polynomial expansions (Cephes coefficients)
    - For |z| <= 2: K_0(z) = -ln(z/2)*I_0(z) + P(z^2 - 2)
    - For |z| > 2: K_0(z) = exp(-z)/sqrt(z) * Q(8/z - 2)
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The modified Bessel function K_0 appears in many contexts:
    - Physics: Yukawa potential, screened Coulomb interactions
    - Electromagnetics: fields of line charges with screening
    - Heat conduction: solutions with exponential decay
    - Statistics: characteristic function of the Laplace distribution

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

       \frac{d}{dz} K_0(z) = -K_1(z)

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} K_0(z) = K_0(z) + \frac{K_1(z)}{z}

    Parameters
    ----------
    z : Tensor
        Input tensor. Must be positive for real tensors.
        Can be complex (with branch cut along negative real axis).

    Returns
    -------
    Tensor
        The modified Bessel function K_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> modified_bessel_k_0(z)
    tensor([0.9244, 0.4210, 0.1139, 0.0347])

    Negative arguments return NaN:

    >>> z = torch.tensor([-1.0, -2.0])
    >>> modified_bessel_k_0(z).isnan()
    tensor([True, True])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 + 0.5j])
    >>> modified_bessel_k_0(z)
    tensor([0.3516-0.2933j, 0.0958-0.0713j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = modified_bessel_k_0(z)
    >>> y.backward()
    >>> z.grad  # equals -K_1(2.0)
    tensor([-0.1399])

    K_0(z) > 0 for all z > 0:

    >>> z = torch.tensor([0.1, 1.0, 10.0])
    >>> (modified_bessel_k_0(z) > 0).all()
    tensor(True)

    .. warning:: Singularity at z = 0

       K_0(z) has a logarithmic singularity at z = 0. As z -> 0+,
       K_0(z) ~ -ln(z/2) - gamma. Computations near z = 0 may have
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
    modified_bessel_k_1 : Modified Bessel function of the second kind of order one
    modified_bessel_i_0 : Modified Bessel function of the first kind of order zero
    modified_bessel_i_1 : Modified Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.modified_bessel_k_0(z)
