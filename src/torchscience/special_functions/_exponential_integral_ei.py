import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def exponential_integral_ei(x: Tensor) -> Tensor:
    r"""
    Exponential integral Ei.

    Computes the exponential integral Ei evaluated at each element of the
    input tensor.

    Mathematical Definition
    -----------------------
    The exponential integral Ei is defined as the Cauchy principal value:

    .. math::

       \mathrm{Ei}(x) = -\int_{-x}^{\infty} \frac{e^{-t}}{t} \, dt
                      = \int_{-\infty}^{x} \frac{e^{t}}{t} \, dt

    For positive real x, Ei(x) can also be expressed via the series:

    .. math::

       \mathrm{Ei}(x) = \gamma + \ln|x| + \sum_{n=1}^{\infty} \frac{x^n}{n \cdot n!}

    where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni constant.

    Special Values
    --------------
    - Ei(0) = -inf (logarithmic singularity)
    - Ei(+inf) = +inf
    - Ei(-inf) = 0
    - Ei(1) approximately equals 1.8951178

    Domain
    ------
    - x: any real or complex value
    - The function has a logarithmic singularity at x = 0

    Algorithm
    ---------
    - For positive x <= 40: Series expansion
    - For positive x > 40: Asymptotic expansion
    - For negative x with |x| <= 1: Series expansion for E1(-x)
    - For negative x with |x| > 1: Continued fraction for E1(-x)

    Applications
    ------------
    The exponential integral appears in many scientific contexts:
    - Radiative transfer and heat conduction
    - Electromagnetism (antenna theory)
    - Quantum mechanics
    - Number theory
    - Probability theory (Gompertz distribution)

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

       \frac{d}{dx} \mathrm{Ei}(x) = \frac{e^x}{x}

    Second-order derivatives are also supported:

    .. math::

       \frac{d^2}{dx^2} \mathrm{Ei}(x) = \frac{(x - 1) e^x}{x^2}

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The exponential integral Ei evaluated at each element of x.
        Output dtype matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> x = torch.tensor([0.5, 1.0, 2.0, 5.0])
    >>> exponential_integral_ei(x)
    tensor([0.4542, 1.8951, 4.9542, 40.1853])

    Negative arguments:

    >>> x = torch.tensor([-0.5, -1.0, -2.0])
    >>> exponential_integral_ei(x)
    tensor([-0.5598, -0.2194, -0.0489])

    Complex input:

    >>> x = torch.tensor([1.0 + 1.0j, 2.0 + 0.5j])
    >>> exponential_integral_ei(x)
    tensor([1.7647+2.3778j, 4.2787+1.6417j])

    Autograd:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = exponential_integral_ei(x)
    >>> y.backward()
    >>> x.grad  # e^2 / 2 = 3.6945...
    tensor([3.6945])

    Relation to E1
    --------------
    The exponential integral Ei is related to E1 (the exponential integral
    of the first kind) by:

    .. math::

       \mathrm{Ei}(x) = -\mathrm{E}_1(-x) \quad \text{for } x > 0

    .. warning:: Singularity at zero

       The function has a logarithmic singularity at x = 0, returning -inf.
       Gradients at x = 0 return NaN.

    See Also
    --------
    scipy.special.expi : SciPy's exponential integral Ei
    """
    return torch.ops.torchscience.exponential_integral_ei(x)
