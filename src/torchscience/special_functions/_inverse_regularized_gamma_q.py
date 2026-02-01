import torch
from torch import Tensor


def inverse_regularized_gamma_q(a: Tensor, y: Tensor) -> Tensor:
    r"""
    Inverse of the regularized upper incomplete gamma function.

    Computes the inverse of the regularized upper incomplete gamma function
    Q(a, x) = y, solving for x given a and y.

    Mathematical Definition
    -----------------------
    Given the regularized upper incomplete gamma function:

    .. math::

        Q(a, x) = \frac{\Gamma(a, x)}{\Gamma(a)} = \frac{1}{\Gamma(a)} \int_x^\infty t^{a-1} e^{-t} dt

    This function finds x such that Q(a, x) = y.

    Since Q(a, x) = 1 - P(a, x), we have:

    .. math::

        Q^{-1}(a, y) = P^{-1}(a, 1 - y)

    Domain
    ------
    - a: positive real numbers (a > 0)
    - y: values in [0, 1]

    Returns
    -------
    - x = infinity when y = 0 (since Q(a, x) -> 0 as x -> infinity)
    - x = 0 when y = 1 (since Q(a, 0) = 1)
    - NaN when a <= 0 or y < 0 or y > 1

    Special Values
    --------------
    - inverse_regularized_gamma_q(a, 0) = infinity for all a > 0
    - inverse_regularized_gamma_q(a, 1) = 0 for all a > 0
    - inverse_regularized_gamma_q(1, y) = -log(y) (exponential distribution)

    Algorithm
    ---------
    Delegates to inverse_regularized_gamma_p via the identity:

    .. math::

        Q^{-1}(a, y) = P^{-1}(a, 1 - y)

    Applications
    ------------
    The inverse regularized upper incomplete gamma function is used in:

    - Computing survival function quantiles
    - Right-tail probability computations
    - Reliability analysis and survival analysis
    - Computing chi-squared critical values for upper-tail tests

    Relation to Other Functions
    ---------------------------
    - scipy.special.gammainccinv: equivalent function in SciPy
    - Complementary chi-squared quantile: For chi-squared with k degrees of freedom,
      upper_quantile(p) = 2 * inverse_regularized_gamma_q(k/2, p)

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common floating-point dtype
    - Supports float32, float64

    Autograd Support
    ----------------
    Gradients are computed via the chain rule and implicit differentiation:

    .. math::

        \frac{\partial x}{\partial y} = -\frac{1}{\partial P/\partial x}

    The negative sign arises from dy_Q = -dy_P.

    Second-order derivatives are supported via numerical differentiation.

    Parameters
    ----------
    a : Tensor
        Shape parameter (must be positive). Broadcasting with y is supported.
    y : Tensor
        Probability value (must be in [0, 1]). Broadcasting with a is supported.

    Returns
    -------
    Tensor
        The value x such that Q(a, x) = y.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0, 2.0, 3.0])
    >>> y = torch.tensor([0.5, 0.5, 0.5])
    >>> inverse_regularized_gamma_q(a, y)
    tensor([0.6931, 1.6783, 2.6741])

    Verify inverse relationship:

    >>> a = torch.tensor([2.0])
    >>> y = torch.tensor([0.3])
    >>> x = inverse_regularized_gamma_q(a, y)
    >>> torch.special.gammaincc(a, x)  # Should be close to y
    tensor([0.3000])

    Relationship with inverse_regularized_gamma_p:

    >>> a = torch.tensor([2.0])
    >>> y = torch.tensor([0.3])
    >>> x_q = inverse_regularized_gamma_q(a, y)
    >>> x_p = inverse_regularized_gamma_p(a, 1 - y)
    >>> torch.allclose(x_q, x_p)
    True

    Autograd:

    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> y = torch.tensor([0.5])
    >>> x = inverse_regularized_gamma_q(a, y)
    >>> x.backward()
    >>> a.grad  # Gradient w.r.t. shape parameter

    .. warning:: Numerical precision

        For y very close to 0 or 1, the result may have reduced precision
        due to the nature of the inverse function.

    .. warning:: Invalid inputs

        The function returns NaN for a <= 0 or y outside [0, 1].

    See Also
    --------
    regularized_gamma_q : The forward function Q(a, x)
    regularized_gamma_p : The complementary function P(a, x) = 1 - Q(a, x)
    inverse_regularized_gamma_p : Inverse of P(a, x)
    gamma : The gamma function
    """
    return torch.ops.torchscience.inverse_regularized_gamma_q(a, y)
