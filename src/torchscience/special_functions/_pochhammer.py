import torch
from torch import Tensor


def pochhammer(z: Tensor, m: Tensor) -> Tensor:
    r"""
    Pochhammer symbol (rising factorial).

    Computes the Pochhammer symbol (z)_m, also known as the rising factorial,
    evaluated at each element.

    Mathematical Definition
    -----------------------
    The Pochhammer symbol is defined as:

    .. math::

       (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}

    For positive integer m, this is equivalent to the product:

    .. math::

       (z)_m = z (z+1) (z+2) \cdots (z+m-1) = \prod_{k=0}^{m-1} (z + k)

    The notation (z)_m is sometimes called the Pochhammer function or
    rising factorial, and is also written as z^{(m)} in some texts.

    Domain
    ------
    - z: any complex number except non-positive integers when m is negative
    - m: any complex number
    - Poles occur when z is a non-positive integer and z+m is not

    Special Values
    --------------
    - (z)_0 = 1 for all z
    - (1)_n = n! for positive integer n
    - (z)_1 = z
    - (a)_n / (b)_n is a ratio of Pochhammer symbols often appearing in
      hypergeometric series
    - (-n)_k = (-1)^k * n! / (n-k)! for non-negative integers n >= k

    Relation to Falling Factorial
    -----------------------------
    The falling factorial (also called descending factorial) is defined as:

    .. math::

       (z)_m^{falling} = z (z-1) (z-2) \cdots (z-m+1)

    The rising factorial is related by:

    .. math::

       (z)_m = (-1)^m (-z)_m^{falling}

    Algorithm
    ---------
    Computed as exp(log_gamma(z+m) - log_gamma(z)) for numerical stability.
    This avoids overflow for large values.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    The Pochhammer symbol appears frequently in:

    - Hypergeometric series coefficients: the generalized hypergeometric
      function pFq is defined using ratios of Pochhammer symbols
    - Binomial coefficients: C(n, k) = (n-k+1)_k / k!
    - Probability distributions (e.g., beta-binomial, negative binomial)
    - Special function identities and recurrence relations
    - Combinatorics and permutation counting

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradients are computed using the digamma function:

    .. math::

       \frac{\partial (z)_m}{\partial z} &= (z)_m \left[ \psi(z+m) - \psi(z) \right] \\
       \frac{\partial (z)_m}{\partial m} &= (z)_m \, \psi(z+m)

    where :math:`\psi(x) = \frac{d}{dx} \ln \Gamma(x)` is the digamma function.

    Second-order derivatives (gradgradcheck) are also supported, using the
    trigamma function for the Hessian computation.

    Parameters
    ----------
    z : Tensor
        Base argument of the Pochhammer symbol. Broadcasting with m is supported.
    m : Tensor
        Exponent argument. The "order" of the rising factorial.
        Broadcasting with z is supported.

    Returns
    -------
    Tensor
        The Pochhammer symbol (z)_m evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage - (z)_0 = 1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> m = torch.tensor([0.0, 0.0, 0.0])
    >>> pochhammer(z, m)
    tensor([1., 1., 1.])

    Rising factorial for (1)_n = n!:

    >>> z = torch.tensor([1.0, 1.0, 1.0, 1.0])
    >>> m = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> pochhammer(z, m)
    tensor([ 1.,  2.,  6., 24.])

    Rising factorial property - z(z+1)...(z+m-1):

    >>> z = torch.tensor([3.0])
    >>> m = torch.tensor([4.0])
    >>> pochhammer(z, m)  # 3 * 4 * 5 * 6 = 360
    tensor([360.])

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> m = torch.tensor([3.0])
    >>> y = pochhammer(z, m)  # (2)_3 = 2*3*4 = 24
    >>> y.backward()
    >>> z.grad
    tensor([26.])  # d/dz (2*3*4) using digamma formula

    .. warning:: Overflow for large arguments

       The Pochhammer symbol can overflow for large m since the product
       grows rapidly. Use log-gamma computations for very large values.

    .. warning:: Poles

       The function returns inf or nan when z is a non-positive integer
       and z+m is not, since these are poles of the gamma function.

    Notes
    -----
    - The implementation uses the identity (z)_m = exp(log_gamma(z+m) - log_gamma(z))
      for numerical stability.
    - For integer m, explicit product formulas may be more accurate for small m,
      but the gamma-based formula is used for generality.

    See Also
    --------
    gamma : Gamma function
    log_gamma : Natural logarithm of the gamma function
    hypergeometric_2_f_1 : Gauss hypergeometric function (uses Pochhammer symbols)
    """
    return torch.ops.torchscience.pochhammer(z, m)
