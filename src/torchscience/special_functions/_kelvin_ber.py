import torch
from torch import Tensor


def kelvin_ber(x: Tensor) -> Tensor:
    r"""
    Kelvin function ber(x).

    Computes the Kelvin function ber(x) evaluated at each element of the
    input tensor. The Kelvin function ber is the real part of the Bessel
    function J_0 at a rotated argument.

    Mathematical Definition
    -----------------------
    The Kelvin function ber(x) is defined as:

    .. math::

       \text{ber}(x) = \text{Re}\left[J_0\left(x \cdot e^{3\pi i/4}\right)\right]

    Equivalently, using the relation:

    .. math::

       \text{ber}(x) + i \cdot \text{bei}(x) = J_0\left(x \cdot e^{3\pi i/4}\right)

    where :math:`e^{3\pi i/4} = \frac{-1 + i}{\sqrt{2}}`.

    Power series expansion (valid for all x):

    .. math::

       \text{ber}(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{((2n)!)^2} \left(\frac{x}{2}\right)^{4n}
                     = 1 - \frac{(x/2)^4}{(2!)^2} + \frac{(x/2)^8}{(4!)^2} - \cdots

    Special Values
    --------------
    - ber(0) = 1
    - ber(x) is an even function: ber(-x) = ber(x)
    - For large x, ber(x) oscillates with exponentially growing amplitude

    Domain
    ------
    - x: any real value
    - The function is defined for complex x as well, computing the same
      analytic continuation

    Algorithm
    ---------
    - For |x| <= 8: Power series expansion with automatic termination
    - For |x| > 8: Asymptotic expansion with exponential factor
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

       \frac{d}{dx} \text{ber}(x) = \frac{\text{ber}_1(x) + \text{bei}_1(x)}{\sqrt{2}}

    where :math:`\text{ber}_1` and :math:`\text{bei}_1` are the order-1 Kelvin functions.

    In practice, the derivative is computed from the power series:

    .. math::

       \text{ber}'(x) = \sum_{n=1}^{\infty} \frac{(-1)^n \cdot 4n}{2 \cdot ((2n)!)^2}
                        \left(\frac{x}{2}\right)^{4n-1}

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Kelvin function ber evaluated at each element of x.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 1.0, 2.0, 5.0])
    >>> kelvin_ber(x)
    tensor([ 1.0000,  0.9844,  0.7517, -6.2301])

    Even function property:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(kelvin_ber(-x), kelvin_ber(x))
    True

    Complex input:

    >>> x = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> kelvin_ber(x)  # doctest: +SKIP

    Autograd:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = kelvin_ber(x)
    >>> y.backward()
    >>> x.grad  # derivative at x=2

    .. warning:: Exponential growth for large arguments

       The Kelvin function ber(x) grows exponentially for large |x|:

       .. math::

          \text{ber}(x) \sim \frac{e^{x/\sqrt{2}}}{\sqrt{2\pi x}}
          \cos\left(\frac{x}{\sqrt{2}} - \frac{\pi}{8}\right)

       For very large arguments, overflow may occur.

    Notes
    -----
    - The Kelvin functions are closely related to Bessel functions.
      Specifically, ber(x) and bei(x) are the real and imaginary parts
      of J_0 evaluated at x * exp(3*pi*i/4).
    - Named after Lord Kelvin who introduced these functions in the
      study of electromagnetic induction.

    See Also
    --------
    bessel_j_0 : Bessel function of the first kind of order zero
    """
    return torch.ops.torchscience.kelvin_ber(x)
