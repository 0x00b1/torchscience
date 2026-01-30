import torch
from torch import Tensor


def kelvin_bei(x: Tensor) -> Tensor:
    r"""
    Kelvin function bei(x).

    Computes the Kelvin function bei(x) evaluated at each element of the
    input tensor. The Kelvin function bei is the imaginary part of the Bessel
    function J_0 at a rotated argument.

    Mathematical Definition
    -----------------------
    The Kelvin function bei(x) is defined as:

    .. math::

       \text{bei}(x) = \text{Im}\left[J_0\left(x \cdot e^{3\pi i/4}\right)\right]

    Equivalently, using the relation:

    .. math::

       \text{ber}(x) + i \cdot \text{bei}(x) = J_0\left(x \cdot e^{3\pi i/4}\right)

    where :math:`e^{3\pi i/4} = \frac{-1 + i}{\sqrt{2}}`.

    Power series expansion (valid for all x):

    .. math::

       \text{bei}(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{((2n+1)!)^2} \left(\frac{x}{2}\right)^{4n+2}
                     = \frac{(x/2)^2}{(1!)^2} - \frac{(x/2)^6}{(3!)^2} + \frac{(x/2)^{10}}{(5!)^2} - \cdots

    Special Values
    --------------
    - bei(0) = 0
    - bei(x) is an even function: bei(-x) = bei(x)
    - For large x, bei(x) oscillates with exponentially growing amplitude

    Domain
    ------
    - x: any real value
    - The function is defined for complex x as well, computing the same
      analytic continuation

    Algorithm
    ---------
    - For |x| <= 20: Power series expansion with automatic termination
    - For |x| > 20: Asymptotic expansion with exponential factor
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
    The gradient is computed using the power series:

    .. math::

       \text{bei}'(x) = \sum_{n=0}^{\infty} \frac{(-1)^n \cdot (2n+1)}{((2n+1)!)^2}
                        \left(\frac{x}{2}\right)^{4n+1}

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Kelvin function bei evaluated at each element of x.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 1.0, 2.0, 5.0])
    >>> kelvin_bei(x)
    tensor([0.0000, 0.2496, 0.9723, 0.1160])

    Even function property:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(kelvin_bei(-x), kelvin_bei(x))
    True

    Complex input:

    >>> x = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> kelvin_bei(x)  # doctest: +SKIP

    Autograd:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = kelvin_bei(x)
    >>> y.backward()
    >>> x.grad  # derivative at x=2

    .. warning:: Exponential growth for large arguments

       The Kelvin function bei(x) grows exponentially for large |x|:

       .. math::

          \text{bei}(x) \sim \frac{e^{x/\sqrt{2}}}{\sqrt{2\pi x}}
          \sin\left(\frac{x}{\sqrt{2}} - \frac{\pi}{8}\right)

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
    kelvin_ber : Kelvin function ber (real part)
    bessel_j_0 : Bessel function of the first kind of order zero
    """
    return torch.ops.torchscience.kelvin_bei(x)
