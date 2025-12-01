import torch
from torch import Tensor


def bessel_y_1(z: Tensor) -> Tensor:
    r"""
    Bessel function of the second kind of order one.

    Computes the Bessel function Y_1(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The Bessel function of the second kind of order one is defined as:

    .. math::

       Y_1(z) = \frac{2}{\pi} \left[ J_1(z) \ln\frac{z}{2} - \frac{1}{z}
                + z \sum_{k=0}^\infty \frac{(-1)^k (H_k + H_{k+1})}{k!(k+1)!}
                \left(\frac{z}{2}\right)^{2k} \right]

    where H_k is the k-th harmonic number and J_1 is the Bessel function
    of the first kind.

    Alternative integral representation:

    .. math::

       Y_1(z) = \frac{2}{\pi} \int_0^\infty e^{-z \sinh t} \cosh t \, dt
                - \frac{2}{\pi z}

    Special Values
    --------------
    - Y_1(0) = -infinity (logarithmic singularity)
    - Y_1(+inf) = 0 (oscillatory decay)
    - Y_1(NaN) = NaN
    - Y_1(z < 0) = NaN (for real z; branch cut along negative real axis)

    Domain
    ------
    - z > 0 for real arguments (Y_1 has a branch cut along z <= 0)
    - For complex z, the principal branch is defined with cut along
      the negative real axis
    - The function has a logarithmic singularity at z = 0

    Algorithm
    ---------
    - Uses rational polynomial approximations (Cephes coefficients)
    - For |z| <= 5: Y_1(z) = z * R(z^2) + (2/pi)[J_1(z)*ln(z/2) - 1/z]
    - For |z| > 5: Asymptotic expansion
      Y_1(z) ~ sqrt(2/(pi*z)) * [P*sin(theta) + Q*cos(theta)]
      where theta = z - 3*pi/4
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Bessel function Y_1 appears in many contexts:
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

       \frac{d}{dz} Y_1(z) = Y_0(z) - \frac{Y_1(z)}{z}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dz^2} Y_1(z) = -Y_1(z) - \frac{Y_0(z)}{z} + \frac{2 Y_1(z)}{z^2}

    Parameters
    ----------
    z : Tensor
        Input tensor. Must be positive for real tensors.
        Can be complex (with branch cut along negative real axis).

    Returns
    -------
    Tensor
        The Bessel function Y_1 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.5, 1.0, 2.0, 3.0])
    >>> bessel_y_1(z)
    tensor([-1.4715, -0.7812, -0.1070,  0.3247])

    Negative arguments return NaN:

    >>> z = torch.tensor([-1.0, -2.0])
    >>> bessel_y_1(z).isnan()
    tensor([True, True])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 + 0.5j])
    >>> bessel_y_1(z)
    tensor([-0.8128-0.4392j, -0.0961-0.2445j])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = bessel_y_1(z)
    >>> y.backward()
    >>> z.grad  # equals Y_0(2.0) - Y_1(2.0)/2.0
    tensor([0.5636])

    First few positive zeros of Y_1: ~2.1971, ~5.4297, ~8.5960

    >>> z = torch.tensor([2.197141])
    >>> bessel_y_1(z).abs() < 1e-4
    tensor([True])

    .. warning:: Singularity at z = 0

       Y_1(z) has a logarithmic singularity at z = 0. As z -> 0+,
       Y_1(z) -> -2/(pi*z). Computations near z = 0 may have
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
    bessel_y_0 : Bessel function of the second kind of order zero
    bessel_j_0 : Bessel function of the first kind of order zero
    bessel_j_1 : Bessel function of the first kind of order one
    """
    return torch.ops.torchscience.bessel_y_1(z)
