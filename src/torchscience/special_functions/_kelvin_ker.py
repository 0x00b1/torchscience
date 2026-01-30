import torch
from torch import Tensor


def kelvin_ker(x: Tensor) -> Tensor:
    r"""
    Kelvin function ker(x).

    Computes the Kelvin function ker(x) evaluated at each element of the
    input tensor. The Kelvin function ker is the real part of the modified
    Bessel function K_0 at a rotated argument.

    Mathematical Definition
    -----------------------
    The Kelvin function ker(x) is defined as:

    .. math::

       \text{ker}(x) = \text{Re}\left[K_0\left(x \cdot e^{i\pi/4}\right)\right]

    Equivalently, using the relation:

    .. math::

       \text{ker}(x) + i \cdot \text{kei}(x) = K_0\left(x \cdot e^{i\pi/4}\right)

    where :math:`e^{i\pi/4} = \frac{1 + i}{\sqrt{2}}`.

    The function can also be expressed using a series involving ber(x) and bei(x):

    .. math::

       \text{ker}(x) = -\left(\ln\frac{x}{2} + \gamma\right) \text{ber}(x)
                       + \frac{\pi}{4} \text{bei}(x) + \sum_{k=0}^{\infty}
                       \frac{(-1)^k H_{2k}}{((2k)!)^2} \left(\frac{x}{2}\right)^{4k}

    where :math:`\gamma` is the Euler-Mascheroni constant and :math:`H_n` is the
    n-th harmonic number.

    Special Values
    --------------
    - ker(0) = +infinity (logarithmic singularity)
    - ker(x) is an even function: ker(-x) = ker(x)
    - ker(x) -> 0 as x -> +infinity (exponential decay)

    Domain
    ------
    - x: any real value (except x = 0 gives infinity)
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

       \frac{d}{dx} \text{ker}(x) = -\frac{1}{x} \text{ber}(x)
                                    - \left(\ln\frac{x}{2} + \gamma\right) \text{ber}'(x)
                                    + \frac{\pi}{4} \text{bei}'(x) + \ldots

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Kelvin function ker evaluated at each element of x.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.5, 1.0, 2.0, 5.0])
    >>> kelvin_ker(x)
    tensor([ 0.8559,  0.2867, -0.0192, -0.0110])

    Even function property:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(kelvin_ker(-x), kelvin_ker(x))
    True

    Complex input:

    >>> x = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> kelvin_ker(x)  # doctest: +SKIP

    Autograd:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = kelvin_ker(x)
    >>> y.backward()
    >>> x.grad  # derivative at x=2

    .. warning:: Singularity at x = 0

       The Kelvin function ker(x) has a logarithmic singularity at x = 0:

       .. math::

          \text{ker}(x) \sim -\ln(x/2) - \gamma \quad \text{as } x \to 0

       This results in +infinity at x = 0.

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
    modified_bessel_k_0 : Modified Bessel function of the second kind K_0
    """
    return torch.ops.torchscience.kelvin_ker(x)
