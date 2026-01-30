import torch
from torch import Tensor


def kelvin_kei(x: Tensor) -> Tensor:
    r"""
    Kelvin function kei(x).

    Computes the Kelvin function kei(x) evaluated at each element of the
    input tensor. The Kelvin function kei is the imaginary part of the modified
    Bessel function K_0 at a rotated argument.

    Mathematical Definition
    -----------------------
    The Kelvin function kei(x) is defined as:

    .. math::

       \text{kei}(x) = \text{Im}\left[K_0\left(x \cdot e^{i\pi/4}\right)\right]

    Equivalently, using the relation:

    .. math::

       \text{ker}(x) + i \cdot \text{kei}(x) = K_0\left(x \cdot e^{i\pi/4}\right)

    where :math:`e^{i\pi/4} = \frac{1 + i}{\sqrt{2}}`.

    The function can also be expressed using a series involving ber(x) and bei(x):

    .. math::

       \text{kei}(x) = -\left(\ln\frac{x}{2} + \gamma\right) \text{bei}(x)
                       - \frac{\pi}{4} \text{ber}(x) + \sum_{k=0}^{\infty}
                       \frac{(-1)^k H_{2k+1}}{((2k+1)!)^2} \left(\frac{x}{2}\right)^{4k+2}

    where :math:`\gamma` is the Euler-Mascheroni constant and :math:`H_n` is the
    n-th harmonic number.

    Special Values
    --------------
    - kei(0) = -pi/4 (approximately -0.7854)
    - kei(x) is an even function: kei(-x) = kei(x)
    - kei(x) -> 0 as x -> +infinity (exponential decay)

    Domain
    ------
    - x: any real value
    - The function is defined for complex x as well, computing the same
      analytic continuation

    Algorithm
    ---------
    - For |x| <= 8: Power series expansion involving ber, bei, and logarithm
    - For |x| > 8: Asymptotic expansion with exponential decay
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The Kelvin functions appear in many physical problems:
    - Electromagnetic waves in conducting media (skin effect)
    - Heat conduction in cylindrical structures
    - Vibrations of thin plates
    - Eddy currents in electrical engineering
    - Solutions to the biharmonic equation

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when x.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dx} \text{kei}(x) = -\frac{1}{x} \text{bei}(x)
                                    - \left(\ln\frac{x}{2} + \gamma\right) \text{bei}'(x)
                                    - \frac{\pi}{4} \text{ber}'(x) + \ldots

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Kelvin function kei evaluated at each element of x.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    >>> kelvin_kei(x)
    tensor([-0.7854, -0.6716, -0.4950, -0.2024,  0.0112])

    Even function property:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(kelvin_kei(-x), kelvin_kei(x))
    True

    Complex input:

    >>> x = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> kelvin_kei(x)  # doctest: +SKIP

    Autograd:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = kelvin_kei(x)
    >>> y.backward()
    >>> x.grad  # derivative at x=2

    Notes
    -----
    - The Kelvin functions are closely related to Bessel functions.
      Specifically, ker(x) and kei(x) are the real and imaginary parts
      of K_0 evaluated at x * exp(i*pi/4).
    - Named after Lord Kelvin who introduced these functions in the
      study of electromagnetic induction.
    - Unlike ber(x) and bei(x) which grow exponentially for large x,
      ker(x) and kei(x) decay exponentially.

    See Also
    --------
    kelvin_ber : Kelvin function ber(x)
    kelvin_bei : Kelvin function bei(x)
    kelvin_ker : Kelvin function ker(x)
    modified_bessel_k_0 : Modified Bessel function of the second kind K_0
    """
    return torch.ops.torchscience.kelvin_kei(x)
