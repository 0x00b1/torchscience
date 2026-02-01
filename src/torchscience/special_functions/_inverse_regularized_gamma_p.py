import torch
from torch import Tensor


def inverse_regularized_gamma_p(a: Tensor, y: Tensor) -> Tensor:
    r"""
    Inverse of the regularized lower incomplete gamma function.

    Computes the inverse of the regularized lower incomplete gamma function
    P(a, x) = y, solving for x given a and y.

    Mathematical Definition
    -----------------------
    Given the regularized lower incomplete gamma function:

    .. math::

        P(a, x) = \frac{\gamma(a, x)}{\Gamma(a)} = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} dt

    This function finds x such that P(a, x) = y.

    Domain
    ------
    - a: positive real numbers (a > 0)
    - y: values in [0, 1]

    Returns
    -------
    - x = 0 when y = 0
    - x = infinity when y = 1
    - NaN when a <= 0 or y < 0 or y > 1

    Special Values
    --------------
    - inverse_regularized_gamma_p(a, 0) = 0 for all a > 0
    - inverse_regularized_gamma_p(a, 1) = infinity for all a > 0
    - inverse_regularized_gamma_p(1, y) = -log(1 - y) (exponential distribution quantile)

    Algorithm
    ---------
    Uses Halley's method with an initial guess based on:

    - Wilson-Hilferty transformation for large a
    - Normal approximation with Cornish-Fisher correction for moderate a
    - Direct series inversion for small a

    The method converges quadratically for most inputs and is robust
    across the entire domain.

    Applications
    ------------
    The inverse regularized gamma function is used in:

    - Computing quantile functions for gamma and chi-squared distributions
    - Statistical hypothesis testing (computing p-values)
    - Monte Carlo simulation (inverse transform sampling)
    - Confidence interval computation

    Relation to Other Functions
    ---------------------------
    - scipy.special.gammaincinv: equivalent function in SciPy
    - Chi-squared quantile: For chi-squared with k degrees of freedom,
      quantile(p) = 2 * inverse_regularized_gamma_p(k/2, p)
    - Gamma distribution quantile: For Gamma(alpha, beta),
      quantile(p) = inverse_regularized_gamma_p(alpha, p) / beta

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common floating-point dtype
    - Supports float32, float64

    Autograd Support
    ----------------
    Gradients are computed via implicit differentiation:

    .. math::

        \frac{\partial x}{\partial y} = \frac{1}{\partial P/\partial x}
            = \frac{\Gamma(a)}{x^{a-1} e^{-x}}

    .. math::

        \frac{\partial x}{\partial a} = -\frac{\partial P/\partial a}{\partial P/\partial x}

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
        The value x such that P(a, x) = y.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0, 2.0, 3.0])
    >>> y = torch.tensor([0.5, 0.5, 0.5])
    >>> inverse_regularized_gamma_p(a, y)
    tensor([0.6931, 1.6783, 2.6741])

    Verify inverse relationship:

    >>> a = torch.tensor([2.0])
    >>> y = torch.tensor([0.3])
    >>> x = inverse_regularized_gamma_p(a, y)
    >>> torch.special.gammainc(a, x)  # Should be close to y
    tensor([0.3000])

    Chi-squared quantile (k=4 degrees of freedom, p=0.95):

    >>> k = 4.0
    >>> p = 0.95
    >>> chi2_quantile = 2 * inverse_regularized_gamma_p(
    ...     torch.tensor([k/2]), torch.tensor([p])
    ... )
    >>> chi2_quantile
    tensor([9.4877])

    Autograd:

    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> y = torch.tensor([0.5])
    >>> x = inverse_regularized_gamma_p(a, y)
    >>> x.backward()
    >>> a.grad  # Gradient w.r.t. shape parameter

    .. warning:: Numerical precision

        For y very close to 0 or 1, the result may have reduced precision
        due to the nature of the inverse function.

    .. warning:: Invalid inputs

        The function returns NaN for a <= 0 or y outside [0, 1].

    See Also
    --------
    regularized_gamma_p : The forward function P(a, x)
    regularized_gamma_q : The complementary function Q(a, x) = 1 - P(a, x)
    inverse_regularized_gamma_q : Inverse of Q(a, x)
    gamma : The gamma function
    """
    return torch.ops.torchscience.inverse_regularized_gamma_p(a, y)
