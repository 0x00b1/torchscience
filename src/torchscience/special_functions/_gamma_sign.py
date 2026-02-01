import torch
from torch import Tensor


def gamma_sign(x: Tensor) -> Tensor:
    r"""
    Sign of the gamma function.

    Returns the sign of the gamma function for real arguments. The gamma
    function is positive for positive real numbers, and alternates sign
    between consecutive poles at non-positive integers.

    Mathematical Definition
    -----------------------
    For real x:

    .. math::

       \text{gammasgn}(x) = \begin{cases}
           +1 & \text{if } \Gamma(x) > 0 \\
           -1 & \text{if } \Gamma(x) < 0 \\
           \text{NaN} & \text{if } x \in \{0, -1, -2, \ldots\}
       \end{cases}

    The sign pattern for negative non-integers is:
    - For :math:`-1 < x < 0`: sign is :math:`-1`
    - For :math:`-2 < x < -1`: sign is :math:`+1`
    - For :math:`-3 < x < -2`: sign is :math:`-1`
    - In general, for :math:`-(n+1) < x < -n`, sign is :math:`(-1)^{n+1}`

    Special Values
    --------------
    - gamma_sign(x) = +1 for all x > 0
    - gamma_sign(x) = NaN for x in {0, -1, -2, -3, ...} (poles of Gamma)
    - gamma_sign(-0.5) = -1
    - gamma_sign(-1.5) = +1
    - gamma_sign(-2.5) = -1

    Domain
    ------
    - x: any real value (not complex)
    - Returns NaN at non-positive integers (poles of gamma)
    - Returns +/-1 everywhere else

    Relation to Gamma Function
    --------------------------
    The gamma sign function is useful for computing log-gamma safely:

    .. math::

       \log|\Gamma(x)| = \log\Gamma(x) \cdot \text{gammasgn}(x)

    This allows working with the absolute value of gamma while tracking
    the sign separately, which is numerically stable for negative arguments.

    Applications
    ------------
    The gamma sign function appears in:
    - Numerical evaluation of ratios of gamma functions
    - Computing log-gamma for negative arguments
    - Stable evaluation of beta functions with negative parameters
    - Statistical distributions involving gamma functions

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Complex tensors are NOT supported (use complex gamma directly)
    - Integer inputs require explicit conversion to floating-point types

    Autograd Support
    ----------------
    Since gamma_sign is piecewise constant (with values +1, -1, or NaN),
    its derivative is zero everywhere it is defined.

    The backward pass returns zero gradients, which is correct because
    infinitesimal changes to the input do not change the sign.

    Parameters
    ----------
    x : Tensor
        Input tensor. Must be real (not complex).

    Returns
    -------
    Tensor
        The sign of the gamma function: +1, -1, or NaN (at poles).
        Output dtype matches input dtype.

    Examples
    --------
    Positive values always have positive gamma:

    >>> x = torch.tensor([0.5, 1.0, 2.0, 3.0, 100.0])
    >>> gamma_sign(x)
    tensor([1., 1., 1., 1., 1.])

    Negative non-integers alternate in sign:

    >>> x = torch.tensor([-0.5, -1.5, -2.5, -3.5])
    >>> gamma_sign(x)
    tensor([-1.,  1., -1.,  1.])

    Poles return NaN:

    >>> x = torch.tensor([0.0, -1.0, -2.0])
    >>> gamma_sign(x)
    tensor([nan, nan, nan])

    Using with log_gamma for stable computation:

    >>> x = torch.tensor([-0.5, -1.5, -2.5])
    >>> log_abs_gamma = torch.lgamma(x.abs())  # log|Gamma(x)|
    >>> sign = gamma_sign(x)  # sign of Gamma(x)
    >>> # Now Gamma(x) = sign * exp(log_abs_gamma)

    Notes
    -----
    - gamma_sign is only defined for real arguments. For complex z,
      the gamma function is generally complex and doesn't have a simple sign.
    - At the poles (non-positive integers), gamma_sign returns NaN since
      the gamma function diverges to +/-infinity depending on the direction
      of approach.
    - This function is equivalent to scipy.special.gammasgn

    See Also
    --------
    gamma : The gamma function
    log_gamma : Natural logarithm of the gamma function
    reciprocal_gamma : Reciprocal of the gamma function
    """
    return torch.ops.torchscience.gamma_sign(x)
