import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def sine_integral_si(x: Tensor) -> Tensor:
    r"""
    Sine integral Si.

    Computes the sine integral Si evaluated at each element of the input tensor.

    Mathematical Definition
    -----------------------
    The sine integral Si is defined as:

    .. math::

       \mathrm{Si}(x) = \int_{0}^{x} \frac{\sin(t)}{t} \, dt

    Si(x) is an odd entire function, meaning it has no singularities in the
    finite complex plane and satisfies Si(-x) = -Si(x).

    Series Expansion
    ----------------
    Si(x) has a convergent Taylor series for all x:

    .. math::

       \mathrm{Si}(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1) \cdot (2n+1)!}
                      = x - \frac{x^3}{18} + \frac{x^5}{600} - \frac{x^7}{35280} + \cdots

    Special Values
    --------------
    - Si(0) = 0
    - Si(+inf) = pi/2 (approximately 1.5708)
    - Si(-inf) = -pi/2 (approximately -1.5708)
    - Si is an odd function: Si(-x) = -Si(x)

    Domain
    ------
    - x: any real or complex value
    - Si is entire, so there are no singularities

    Derivatives
    -----------
    The derivative of Si(x) is the sinc function:

    .. math::

       \frac{d}{dx} \mathrm{Si}(x) = \frac{\sin(x)}{x}

    At x = 0, the derivative has a removable singularity with value 1:

    .. math::

       \lim_{x \to 0} \frac{\sin(x)}{x} = 1

    The second derivative is:

    .. math::

       \frac{d^2}{dx^2} \mathrm{Si}(x) = \frac{x \cos(x) - \sin(x)}{x^2}

    which equals 0 at x = 0.

    Applications
    ------------
    The sine integral appears in:
    - Signal processing (Gibbs phenomenon analysis)
    - Diffraction theory
    - Antenna theory
    - Heat conduction problems
    - Fourier analysis

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs must be explicitly converted to floating-point types

    Autograd Support
    ----------------
    Full autograd support including second-order derivatives.

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The sine integral Si evaluated at each element of x. Output dtype
        matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    >>> sine_integral_si(x)
    tensor([0.0000, 0.4931, 0.9461, 1.6054, 1.5499])

    Approaching asymptotic value:

    >>> x = torch.tensor([10.0, 20.0, 50.0])
    >>> sine_integral_si(x)  # approaches pi/2 ~ 1.5708
    tensor([1.6583, 1.5482, 1.5516])

    Odd function property:

    >>> x = torch.tensor([1.0, -1.0])
    >>> sine_integral_si(x)
    tensor([ 0.9461, -0.9461])

    Complex input:

    >>> x = torch.tensor([1.0 + 1.0j])
    >>> sine_integral_si(x)
    tensor([1.1042+0.8825j])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = sine_integral_si(x)
    >>> y.backward()
    >>> x.grad  # sin(1)/1 = 0.8415
    tensor([0.8415])

    The function is smooth at x = 0:

    >>> x = torch.tensor([0.0], requires_grad=True)
    >>> y = sine_integral_si(x)
    >>> y.backward()
    >>> x.grad  # derivative at 0 is 1
    tensor([1.])

    See Also
    --------
    exponential_integral_ein : Complementary exponential integral Ein
    """
    return torch.ops.torchscience.sine_integral_si(x)
