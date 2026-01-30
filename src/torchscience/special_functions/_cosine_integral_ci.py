import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def cosine_integral_ci(x: Tensor) -> Tensor:
    r"""
    Cosine integral Ci.

    Computes the cosine integral Ci evaluated at each element of the input tensor.

    Mathematical Definition
    -----------------------
    The cosine integral Ci is defined as:

    .. math::

       \mathrm{Ci}(x) = \gamma + \ln(x) + \int_{0}^{x} \frac{\cos(t) - 1}{t} \, dt

    where :math:`\gamma \approx 0.5772156649` is the Euler-Mascheroni constant.

    Equivalently, it can be defined as:

    .. math::

       \mathrm{Ci}(x) = -\int_{x}^{\infty} \frac{\cos(t)}{t} \, dt

    Series Expansion
    ----------------
    Ci(x) has a series expansion:

    .. math::

       \mathrm{Ci}(x) = \gamma + \ln(x) + \sum_{n=1}^{\infty} \frac{(-1)^n x^{2n}}{2n \cdot (2n)!}
                      = \gamma + \ln(x) - \frac{x^2}{4} + \frac{x^4}{96} - \frac{x^6}{4320} + \cdots

    Special Values
    --------------
    - Ci(0) = -inf (logarithmic singularity)
    - Ci(+inf) = 0 (approaches 0 with oscillation)
    - Ci is only defined for x > 0 (real positive values)

    Domain
    ------
    - x > 0: Ci is well-defined and real
    - x <= 0: Returns NaN (function is not defined for non-positive real values)
    - For complex z with Re(z) > 0 or Im(z) != 0: Ci(z) is defined via analytic continuation

    Derivatives
    -----------
    The derivative of Ci(x) is:

    .. math::

       \frac{d}{dx} \mathrm{Ci}(x) = \frac{\cos(x)}{x}

    Note that this has no removable singularity at x = 0 (unlike Si).

    The second derivative is:

    .. math::

       \frac{d^2}{dx^2} \mathrm{Ci}(x) = \frac{-x \sin(x) - \cos(x)}{x^2}

    Applications
    ------------
    The cosine integral appears in:
    - Electromagnetic wave propagation
    - Antenna theory (radiation patterns)
    - Signal processing
    - Diffraction theory
    - Heat transfer problems
    - Quantum mechanics

    Relation to Other Functions
    ---------------------------
    The cosine and sine integrals are related to the exponential integral:

    .. math::

       \mathrm{Ci}(x) + i \cdot \mathrm{Si}(x) = \mathrm{Ei}(ix) + \frac{i\pi}{2}

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs must be explicitly converted to floating-point types

    Autograd Support
    ----------------
    Full autograd support including second-order derivatives for x > 0.
    Gradients at x <= 0 are NaN since Ci is not defined there.

    Parameters
    ----------
    x : Tensor
        Input tensor. Must contain positive values for real inputs.
        Can be floating-point or complex.

    Returns
    -------
    Tensor
        The cosine integral Ci evaluated at each element of x. Output dtype
        matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> x = torch.tensor([0.5, 1.0, 2.0, 5.0])
    >>> cosine_integral_ci(x)
    tensor([-0.1778, 0.3374, 0.4230, -0.1900])

    Larger values approach 0:

    >>> x = torch.tensor([10.0, 20.0, 50.0])
    >>> cosine_integral_ci(x)  # approaches 0 with oscillation
    tensor([-0.0455, 0.0444, -0.0056])

    Behavior near singularity:

    >>> x = torch.tensor([0.001, 0.01, 0.1])
    >>> cosine_integral_ci(x)  # approaches -inf as x -> 0
    tensor([-6.3315, -4.0280, -1.7279])

    Non-positive values return NaN:

    >>> x = torch.tensor([-1.0, 0.0])
    >>> cosine_integral_ci(x)
    tensor([nan, nan])

    Complex input:

    >>> x = torch.tensor([1.0 + 1.0j])
    >>> cosine_integral_ci(x)
    tensor([0.8822+0.2874j])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = cosine_integral_ci(x)
    >>> y.backward()
    >>> x.grad  # cos(1)/1 = 0.5403
    tensor([0.5403])

    See Also
    --------
    sine_integral_si : Sine integral Si
    exponential_integral_ei : Exponential integral Ei
    """
    return torch.ops.torchscience.cosine_integral_ci(x)
