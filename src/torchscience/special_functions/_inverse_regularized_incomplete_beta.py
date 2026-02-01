import torch
from torch import Tensor


def inverse_regularized_incomplete_beta(
    a: Tensor, b: Tensor, y: Tensor
) -> Tensor:
    r"""
    Inverse of the regularized incomplete beta function.

    Computes the inverse of the regularized incomplete beta function
    I_x(a, b) = y, solving for x given a, b, and y.

    Mathematical Definition
    -----------------------
    Given the regularized incomplete beta function:

    .. math::

        I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}
                  = \frac{1}{B(a, b)} \int_0^x t^{a-1} (1-t)^{b-1} dt

    This function finds x such that I_x(a, b) = y.

    Domain
    ------
    - a: positive real numbers (a > 0)
    - b: positive real numbers (b > 0)
    - y: values in [0, 1]

    Returns
    -------
    - x = 0 when y = 0
    - x = 1 when y = 1
    - NaN when a <= 0 or b <= 0 or y < 0 or y > 1

    Special Values
    --------------
    - inverse_regularized_incomplete_beta(a, b, 0) = 0
    - inverse_regularized_incomplete_beta(a, b, 1) = 1
    - inverse_regularized_incomplete_beta(1, 1, y) = y (uniform distribution)
    - inverse_regularized_incomplete_beta(1, b, y) = 1 - (1-y)^(1/b)
    - inverse_regularized_incomplete_beta(a, 1, y) = y^(1/a)

    Algorithm
    ---------
    Uses Halley's method with an initial guess based on:

    - Normal approximation using Wilson-Hilferty transformation
    - Power law refinement for small a or b
    - Symmetry exploitation for large y

    The method converges cubically for most inputs and is robust
    across the entire domain.

    Applications
    ------------
    The inverse regularized incomplete beta function is used in:

    - Computing quantile functions for beta, binomial, and F distributions
    - Statistical hypothesis testing (computing p-values)
    - Bayesian credible interval computation
    - Monte Carlo simulation (inverse transform sampling)
    - A/B testing analysis

    Relation to Other Functions
    ---------------------------
    - scipy.special.betaincinv: equivalent function in SciPy
    - Beta distribution quantile: For Beta(alpha, beta),
      quantile(p) = inverse_regularized_incomplete_beta(alpha, beta, p)
    - F distribution quantile: For F(d1, d2),
      quantile(p) = (d2/d1) * inverse_regularized_incomplete_beta(d1/2, d2/2, p) /
                    (1 - inverse_regularized_incomplete_beta(d1/2, d2/2, p))
    - Binomial distribution: Related via the regularized incomplete beta

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common floating-point dtype
    - Supports float32, float64

    Autograd Support
    ----------------
    Gradients are computed via implicit differentiation:

    .. math::

        \frac{\partial x}{\partial y} = \frac{1}{\partial I/\partial x}
            = \frac{B(a, b)}{x^{a-1} (1-x)^{b-1}}

    .. math::

        \frac{\partial x}{\partial a} = -\frac{\partial I/\partial a}{\partial I/\partial x}

    .. math::

        \frac{\partial x}{\partial b} = -\frac{\partial I/\partial b}{\partial I/\partial x}

    Second-order derivatives are supported via numerical differentiation.

    Parameters
    ----------
    a : Tensor
        First shape parameter (must be positive). Broadcasting supported.
    b : Tensor
        Second shape parameter (must be positive). Broadcasting supported.
    y : Tensor
        Probability value (must be in [0, 1]). Broadcasting supported.

    Returns
    -------
    Tensor
        The value x such that I_x(a, b) = y.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([2.0, 3.0, 5.0])
    >>> b = torch.tensor([3.0, 3.0, 2.0])
    >>> y = torch.tensor([0.5, 0.5, 0.5])
    >>> inverse_regularized_incomplete_beta(a, b, y)
    tensor([0.3857, 0.5000, 0.6860])

    Verify inverse relationship:

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> y = torch.tensor([0.3])
    >>> x = inverse_regularized_incomplete_beta(a, b, y)
    >>> torch.special.betainc(a, b, x)  # Should be close to y
    tensor([0.3000])

    Beta distribution median (a=b case gives x=0.5):

    >>> a = torch.tensor([5.0])
    >>> b = torch.tensor([5.0])
    >>> inverse_regularized_incomplete_beta(a, b, torch.tensor([0.5]))
    tensor([0.5000])

    Autograd:

    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> b = torch.tensor([3.0])
    >>> y = torch.tensor([0.5])
    >>> x = inverse_regularized_incomplete_beta(a, b, y)
    >>> x.backward()
    >>> a.grad  # Gradient w.r.t. first shape parameter

    .. warning:: Numerical precision

        For y very close to 0 or 1, or when a or b are very small,
        the result may have reduced precision due to the nature of
        the inverse function.

    .. warning:: Invalid inputs

        The function returns NaN for a <= 0, b <= 0, or y outside [0, 1].

    See Also
    --------
    incomplete_beta : The forward function I_x(a, b)
    beta : The beta function B(a, b)
    log_beta : The logarithm of the beta function
    """
    return torch.ops.torchscience.inverse_regularized_incomplete_beta(a, b, y)
