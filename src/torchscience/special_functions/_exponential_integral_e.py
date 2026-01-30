import torch
from torch import Tensor


def exponential_integral_e(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Generalized exponential integral function of order n.

    Computes the generalized exponential integral :math:`E_n(x)` evaluated at
    each element of the input tensors, where n is the order and x is the argument.

    Mathematical Definition
    -----------------------
    The generalized exponential integral of order n is defined as:

    .. math::

       E_n(x) = \int_1^{\infty} \frac{e^{-xt}}{t^n} dt

    for :math:`x > 0` and integer :math:`n \geq 0`.

    The function can also be expressed in terms of the incomplete gamma function:

    .. math::

       E_n(x) = x^{n-1} \Gamma(1-n, x)

    Special Cases
    -------------
    - :math:`E_0(x) = \frac{e^{-x}}{x}`
    - :math:`E_1(x)` is the standard exponential integral (see `exponential_integral_e_1`)
    - For :math:`n \geq 2`: :math:`E_n(x) = \frac{e^{-x} - x \cdot E_{n-1}(x)}{n-1}`

    Special Values
    --------------
    - :math:`E_n(0) = \frac{1}{n-1}` for :math:`n \geq 2`
    - :math:`E_0(0) = +\infty`
    - :math:`E_1(0) = +\infty`
    - :math:`E_n(+\infty) = 0`
    - :math:`E_n(x < 0) = \text{NaN}` for real inputs

    Recurrence Relation
    -------------------
    The generalized exponential integrals satisfy the recurrence:

    .. math::

       E_n(x) = \frac{e^{-x} - x \cdot E_{n-1}(x)}{n-1} \quad \text{for } n \geq 2

    Derivative
    ----------
    The derivative with respect to x is:

    .. math::

       \frac{d}{dx} E_n(x) = -E_{n-1}(x)

    Domain
    ------
    - n: non-negative integer (0, 1, 2, 3, ...)
    - x: positive real number (x > 0)
    - For x = 0: E_n(0) = 1/(n-1) for n >= 2, infinity for n = 0, 1
    - For x < 0: returns NaN

    Algorithm
    ---------
    - For n = 0: Direct formula :math:`E_0(x) = e^{-x}/x`
    - For n = 1: Uses `exponential_integral_e_1`
    - For n >= 2 with small x: Upward recurrence from E_1
    - For n >= 2 with large x: Continued fraction expansion
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Applications
    ------------
    The generalized exponential integral appears in many scientific contexts:
    - Radiative transfer and atmospheric physics
    - Heat conduction and diffusion problems
    - Neutron transport theory
    - Chemical kinetics
    - Probability theory (related to gamma distribution)
    - Quantum mechanics

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported for x when it requires grad.
    The gradient with respect to n is always zero since n must be a discrete
    non-negative integer.

    .. math::

       \frac{\partial}{\partial x} E_n(x) = -E_{n-1}(x)

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the exponential integral. Must be a non-negative integer.
        Broadcasting with x is supported.
    x : Tensor
        Argument at which to evaluate the exponential integral.
        Must be non-negative for real inputs.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The generalized exponential integral E_n(x) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> x = torch.tensor([1.0, 1.0, 1.0, 1.0])
    >>> exponential_integral_e(n, x)
    tensor([0.3679, 0.2194, 0.1485, 0.1097])

    Matches specialized E_1 function:

    >>> x = torch.tensor([0.5, 1.0, 2.0, 5.0])
    >>> n = torch.ones_like(x)
    >>> torch.allclose(exponential_integral_e(n, x), exponential_integral_e_1(x))
    True

    Value at x=0 for n >= 2:

    >>> n = torch.tensor([2.0, 3.0, 4.0, 5.0])
    >>> x = torch.tensor([0.0, 0.0, 0.0, 0.0])
    >>> exponential_integral_e(n, x)
    tensor([1.0000, 0.5000, 0.3333, 0.2500])

    Recurrence relation verification:

    >>> n = torch.tensor([3.0])
    >>> x = torch.tensor([2.0])
    >>> E_n = exponential_integral_e(n, x)
    >>> E_nm1 = exponential_integral_e(n - 1, x)
    >>> expected = (torch.exp(-x) - x * E_nm1) / (n - 1)
    >>> torch.allclose(E_n, expected)
    True

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> x = torch.tensor([0.5, 1.0, 2.0])        # (3,)
    >>> exponential_integral_e(n, x).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([2.0])
    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = exponential_integral_e(n, x)
    >>> y.backward()
    >>> x.grad  # equals -E_1(x)
    tensor([-0.2194])

    Derivative identity verification:

    >>> n = torch.tensor([3.0])
    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = exponential_integral_e(n, x)
    >>> grad = torch.autograd.grad(y, x)[0]
    >>> expected_grad = -exponential_integral_e(n - 1, x.detach())
    >>> torch.allclose(grad, expected_grad)
    True

    .. warning:: Discrete parameter

       The order n must be a non-negative integer. Non-integer values of n
       will result in NaN. The gradient with respect to n is always zero.

    Notes
    -----
    - For n = 1, the specialized function `exponential_integral_e_1` is used
      internally for better accuracy.
    - The generalized exponential integrals are related to the incomplete
      gamma function by :math:`E_n(x) = x^{n-1} \Gamma(1-n, x)`.
    - For large x, an asymptotic expansion is available:
      :math:`E_n(x) \sim \frac{e^{-x}}{x}(1 - n/x + n(n+1)/x^2 - \cdots)`

    See Also
    --------
    exponential_integral_e_1 : Exponential integral E_1
    exponential_integral_ei : Exponential integral Ei
    exponential_integral_ein : Complementary exponential integral Ein
    """
    return torch.ops.torchscience.exponential_integral_e(n, x)
