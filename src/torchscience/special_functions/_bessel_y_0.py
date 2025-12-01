import torch
from torch import Tensor


def bessel_y_0(z: Tensor) -> Tensor:
    r"""
    Bessel function of the second kind of order zero.

    Computes the Bessel function Y_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The Bessel function of the second kind of order zero is defined as:

    .. math::

       Y_0(z) = \frac{2}{\pi} \left[ J_0(z) \left( \ln\frac{z}{2} + \gamma \right)
                + \sum_{k=1}^\infty \frac{(-1)^{k+1} H_k}{(k!)^2}
                \left(\frac{z}{2}\right)^{2k} \right]

    where gamma is the Euler-Mascheroni constant, H_k is the k-th harmonic
    number, and J_0 is the Bessel function of the first kind.

    Alternative integral representation:

    .. math::

       Y_0(z) = \frac{2}{\pi} \int_0^\infty \cos(z \cosh t) \, dt

    Special Values
    --------------
    - Y_0(0) = -infinity (logarithmic singularity)
    - Y_0(+inf) = 0 (oscillatory decay)
    - Y_0(NaN) = NaN
    - Y_0(z < 0) = NaN (for real z; branch cut along negative real axis)

    Domain
    ------
    - z > 0 for real arguments (Y_0 has a branch cut along z <= 0)
    - For complex z, the principal branch is defined with cut along
      the negative real axis
    - The function has a logarithmic singularity at z = 0

    Algorithm
    ---------
    - Uses rational polynomial approximations (Cephes coefficients)
    - For |z| <= 5: Y_0(z) = R(z^2) + (2/pi) * J_0(z) * ln(z)
    - For |z| > 5: Asymptotic expansion
      Y_0(z) ~ sqrt(2/(pi*z)) * [P*sin(theta) + Q*cos(theta)]
      where theta = z - pi/4
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Bessel function Y_0 appears in many contexts:
    - Physics: cylindrical waveguides (Neumann functions)
    - Electromagnetics: field solutions in cylindrical coordinates
    - Acoustics: cylindrical wave propagation
    - Heat conduction: solutions to the heat equation in cylinders

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

       \frac{d}{dz} Y_0(z) = -Y_1(z)

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} Y_0(z) = -Y_0(z) + \frac{Y_1(z)}{z}

    Parameters
    ----------
    z : Tensor
        Input tensor. Must be positive for real tensors.
        Can be complex (with branch cut along negative real axis).

    Returns
    -------
    Tensor
        The Bessel function Y_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> bessel_y_0(z)
    tensor([-0.4445,  0.0883,  0.5104,  0.3769])

    Negative arguments return NaN:

    >>> z = torch.tensor([-1.0, -2.0])
    >>> bessel_y_0(z).isnan()
    tensor([True, True])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 + 0.5j])
    >>> bessel_y_0(z)
    tensor([0.1422-0.3855j, 0.5369-0.1712j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = bessel_y_0(z)
    >>> y.backward()
    >>> z.grad  # equals -Y_1(2.0)
    tensor([0.1070])

    First few positive zeros of Y_0: ~0.8936, ~3.9577, ~7.0861

    >>> z = torch.tensor([0.893577])
    >>> bessel_y_0(z).abs() < 1e-4
    tensor([True])

    .. warning:: Singularity at z = 0

       Y_0(z) has a logarithmic singularity at z = 0. As z -> 0+,
       Y_0(z) -> (2/pi) * ln(z/2). Computations near z = 0 may have
       reduced precision.

    Notes
    -----
    - Complex accuracy: The Cephes approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.
    - The implementation uses the Cephes library coefficients (public domain).
    - For z on the negative real axis, the function returns NaN for real
      inputs. For complex inputs, the principal branch is used.

    See Also
    --------
    bessel_y_1 : Bessel function of the second kind of order one
    bessel_j_0 : Bessel function of the first kind of order zero
    bessel_j_1 : Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.bessel_y_0(z)
