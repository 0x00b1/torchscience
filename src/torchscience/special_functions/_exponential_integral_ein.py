import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def exponential_integral_ein(x: Tensor) -> Tensor:
    r"""
    Complementary exponential integral Ein.

    Computes the complementary exponential integral Ein evaluated at each
    element of the input tensor.

    Mathematical Definition
    -----------------------
    The complementary exponential integral Ein is defined as:

    .. math::

       \mathrm{Ein}(x) = \int_{0}^{x} \frac{1 - e^{-t}}{t} \, dt

    Ein(x) is an entire function, meaning it has no singularities in the
    finite complex plane. This contrasts with Ei(x) which has a logarithmic
    singularity at x = 0.

    Series Expansion
    ----------------
    Ein(x) has a convergent Taylor series for all x:

    .. math::

       \mathrm{Ein}(x) = \sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n \cdot n!}
                       = x - \frac{x^2}{4} + \frac{x^3}{18} - \frac{x^4}{96} + \cdots

    Relation to Ei
    --------------
    For real x > 0:

    .. math::

       \mathrm{Ein}(x) = \mathrm{Ei}(x) - \gamma - \ln|x|

    where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni constant.

    Special Values
    --------------
    - Ein(0) = 0
    - Ein(+inf) = +inf
    - Ein(-inf) = -inf

    The function is smooth at x = 0 (unlike Ei which has a singularity).

    Domain
    ------
    - x: any real or complex value
    - Ein is entire, so there are no singularities

    Derivatives
    -----------
    The derivative of Ein(x) is:

    .. math::

       \frac{d}{dx} \mathrm{Ein}(x) = \frac{1 - e^{-x}}{x}

    At x = 0, the derivative has a removable singularity with value 1:

    .. math::

       \lim_{x \to 0} \frac{1 - e^{-x}}{x} = 1

    The second derivative is:

    .. math::

       \frac{d^2}{dx^2} \mathrm{Ein}(x) = \frac{e^{-x}(x + 1) - 1}{x^2}

    Applications
    ------------
    The complementary exponential integral Ein appears in:
    - Analysis of entire functions
    - Numerical computation of Ei (since Ein avoids the singularity)
    - Heat conduction problems
    - Radiation transfer

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
        The complementary exponential integral Ein evaluated at each element
        of x. Output dtype matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    >>> exponential_integral_ein(x)
    tensor([0.0000, 0.4438, 0.7966, 1.3193, 2.1878])

    Negative arguments:

    >>> x = torch.tensor([-0.5, -1.0, -2.0])
    >>> exponential_integral_ein(x)
    tensor([-0.5702, -1.3179, -3.6839])

    Complex input:

    >>> x = torch.tensor([1.0 + 1.0j, 2.0 + 0.5j])
    >>> exponential_integral_ein(x)
    tensor([0.9241+0.6061j, 1.3377+0.2145j])

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = exponential_integral_ein(x)
    >>> y.backward()
    >>> x.grad  # (1 - e^(-1)) / 1 = 0.6321...
    tensor([0.6321])

    The function is smooth at x = 0:

    >>> x = torch.tensor([0.0], requires_grad=True)
    >>> y = exponential_integral_ein(x)
    >>> y.backward()
    >>> x.grad  # derivative at 0 is 1
    tensor([1.])

    See Also
    --------
    exponential_integral_ei : Exponential integral Ei
    exponential_integral_e_1 : Exponential integral E1
    """
    return torch.ops.torchscience.exponential_integral_ein(x)
