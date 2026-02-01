import torch
from torch import Tensor


def inverse_complementary_regularized_incomplete_beta(
    a: Tensor, b: Tensor, y: Tensor
) -> Tensor:
    r"""
    Inverse of the complementary regularized incomplete beta function.

    Computes the inverse of the complementary regularized incomplete beta function
    I_c(a, b, x) = 1 - I_x(a, b) = y, solving for x given a, b, and y.

    Mathematical Definition
    -----------------------
    Given the complementary regularized incomplete beta function:

    .. math::

        I_c(a, b, x) = 1 - I_x(a, b) = 1 - \frac{B(x; a, b)}{B(a, b)}

    This function finds x such that I_c(a, b, x) = y, which is equivalent to:

    .. math::

        x = I^{-1}(a, b, 1 - y)

    Domain
    ------
    - a: positive real numbers (a > 0)
    - b: positive real numbers (b > 0)
    - y: values in [0, 1]

    Returns
    -------
    - x = 1 when y = 0 (since I_c(a, b, 1) = 0)
    - x = 0 when y = 1 (since I_c(a, b, 0) = 1)
    - NaN when a <= 0 or b <= 0 or y < 0 or y > 1

    Special Values
    --------------
    - inverse_complementary_regularized_incomplete_beta(a, b, 0) = 1
    - inverse_complementary_regularized_incomplete_beta(a, b, 1) = 0

    Algorithm
    ---------
    Delegates to inverse_regularized_incomplete_beta via:

    .. math::

        I_c^{-1}(a, b, y) = I^{-1}(a, b, 1 - y)

    Applications
    ------------
    The inverse complementary regularized incomplete beta function is used in:

    - Computing upper-tail quantiles for beta distributions
    - Survival function analysis
    - Right-tail probability computations
    - F distribution upper quantiles

    Relation to Other Functions
    ---------------------------
    - scipy.special.betainccinv: equivalent function in SciPy

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common floating-point dtype
    - Supports float32, float64

    Autograd Support
    ----------------
    Gradients are computed via the chain rule and implicit differentiation.
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
        The value x such that 1 - I_x(a, b) = y.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([2.0, 3.0, 5.0])
    >>> b = torch.tensor([3.0, 3.0, 2.0])
    >>> y = torch.tensor([0.5, 0.5, 0.5])
    >>> inverse_complementary_regularized_incomplete_beta(a, b, y)

    Relationship with inverse_regularized_incomplete_beta:

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> y = torch.tensor([0.3])
    >>> x_c = inverse_complementary_regularized_incomplete_beta(a, b, y)
    >>> x_p = inverse_regularized_incomplete_beta(a, b, 1 - y)
    >>> torch.allclose(x_c, x_p)
    True

    Autograd:

    >>> a = torch.tensor([2.0], requires_grad=True)
    >>> b = torch.tensor([3.0])
    >>> y = torch.tensor([0.5])
    >>> x = inverse_complementary_regularized_incomplete_beta(a, b, y)
    >>> x.backward()
    >>> a.grad

    .. warning:: Numerical precision

        For y very close to 0 or 1, the result may have reduced precision.

    .. warning:: Invalid inputs

        The function returns NaN for a <= 0, b <= 0, or y outside [0, 1].

    See Also
    --------
    inverse_regularized_incomplete_beta : Inverse of I_x(a, b)
    incomplete_beta : The forward function I_x(a, b)
    beta : The beta function B(a, b)
    """
    return torch.ops.torchscience.inverse_complementary_regularized_incomplete_beta(
        a, b, y
    )
