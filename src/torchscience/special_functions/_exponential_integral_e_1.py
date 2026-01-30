import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def exponential_integral_e_1(x: Tensor) -> Tensor:
    r"""
    Exponential integral E_1.

    Computes the exponential integral E_1 evaluated at each element of the
    input tensor.

    Mathematical Definition
    -----------------------
    The exponential integral E_1 is defined for positive real x as:

    .. math::

       E_1(x) = \int_x^{\infty} \frac{e^{-t}}{t} \, dt

    For the series representation (valid for x > 0):

    .. math::

       E_1(x) = -\gamma - \ln(x) - \sum_{n=1}^{\infty} \frac{(-1)^n x^n}{n \cdot n!}

    where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni constant.

    Special Values
    --------------
    - E_1(0) = +inf (logarithmic singularity)
    - E_1(+inf) = 0
    - E_1(x) is undefined for x < 0 for real inputs (returns NaN)

    Relation to Ei
    --------------
    For x > 0:

    .. math::

       E_1(x) = -\mathrm{Ei}(-x)

    Relation to incomplete gamma:

    .. math::

       E_1(x) = \Gamma(0, x)

    Domain
    ------
    - Real x: x > 0 (undefined for x <= 0)
    - Complex z: all values except z = 0

    Algorithm
    ---------
    - For 0 < x <= 1: Series expansion
    - For x > 1: Continued fraction for better convergence

    Applications
    ------------
    The exponential integral E_1 appears in many scientific contexts:
    - Radiative transfer and heat conduction
    - Electromagnetism (antenna theory)
    - Hydrology (groundwater flow, Theis equation)
    - Reactor physics (neutron transport)
    - Probability theory

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs must be explicitly converted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when x.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dx} E_1(x) = -\frac{e^{-x}}{x}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dx^2} E_1(x) = \frac{e^{-x}(x + 1)}{x^2}

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.
        For real inputs, must be positive (x > 0).

    Returns
    -------
    Tensor
        The exponential integral E_1 evaluated at each element of x.
        Output dtype matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> x = torch.tensor([0.5, 1.0, 2.0, 5.0])
    >>> exponential_integral_e_1(x)
    tensor([0.5598, 0.2194, 0.0489, 0.0011])

    Autograd:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = exponential_integral_e_1(x)
    >>> y.backward()
    >>> x.grad  # -e^{-2} / 2 = -0.0677...
    tensor([-0.0677])

    Complex input:

    >>> z = torch.tensor([1.0 + 1.0j, 2.0 + 0.5j])
    >>> exponential_integral_e_1(z)
    tensor([0.0003-0.3379j, 0.0326-0.0146j])

    Relation to Ei:

    >>> x = torch.tensor([1.0, 2.0])
    >>> e1 = exponential_integral_e_1(x)
    >>> ei_neg = exponential_integral_ei(-x)  # Ei(-x)
    >>> torch.allclose(-ei_neg, e1, atol=1e-6)  # E_1(x) = -Ei(-x)
    True

    .. warning:: Singularity at zero

       The function has a logarithmic singularity at x = 0, returning +inf.
       Gradients at x = 0 return NaN.

    .. warning:: Undefined for negative real inputs

       For real inputs x < 0, the function returns NaN. Use complex inputs
       to evaluate E_1 on the negative real axis.

    See Also
    --------
    exponential_integral_ei : Exponential integral Ei
    scipy.special.exp1 : SciPy's exponential integral E_1
    """
    return torch.ops.torchscience.exponential_integral_e_1(x)
