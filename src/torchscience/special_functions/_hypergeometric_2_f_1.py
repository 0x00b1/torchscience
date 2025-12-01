import torch
from torch import Tensor


def hypergeometric_2_f_1(a: Tensor, b: Tensor, c: Tensor, z: Tensor) -> Tensor:
    r"""
    Gauss hypergeometric function 2F1(a, b; c; z).

    Computes the Gauss hypergeometric function, one of the most important
    special functions in mathematics.

    Mathematical Definition
    -----------------------
    The Gauss hypergeometric function is defined by the series:

    .. math::

       {}_2F_1(a, b; c; z) = \sum_{n=0}^{\infty} \frac{(a)_n (b)_n}{(c)_n \, n!} z^n

    where :math:`(x)_n = x(x+1)\cdots(x+n-1)` is the Pochhammer symbol
    (rising factorial), with :math:`(x)_0 = 1`.

    Domain
    ------
    - a, b: any real or complex values
    - c: must not be a non-positive integer (poles at c = 0, -1, -2, ...)
    - z: any real or complex value (analytic continuation handles ``|z| >= 1``)

    Convergence of the defining series:

    - Absolutely convergent for ``|z| < 1``
    - Conditionally convergent for ``|z| = 1`` if Re(c - a - b) > 0
    - Divergent for ``|z| > 1`` (requires analytic continuation)

    For ``|z| >= 1``, the function uses a linear transformation (DLMF 15.8.2)
    to analytically continue the result.

    Special Values
    --------------
    - 2F1(a, b; c; 0) = 1
    - 2F1(0, b; c; z) = 1
    - 2F1(a, 0; c; z) = 1
    - 2F1(a, b; b; z) = (1-z)^(-a)
    - 2F1(1, 1; 2; z) = -log(1-z)/z
    - 2F1(1/2, 1; 3/2; z^2) = (1/2z) * log((1+z)/(1-z)) = arctanh(z)/z

    Applications
    ------------
    The Gauss hypergeometric function appears in many contexts:
    - Regularized incomplete beta function: I_z(a,b) = z^a / (a*B(a,b)) * 2F1(a, 1-b; a+1; z)
    - Legendre functions and spherical harmonics
    - Jacobi, Gegenbauer, and Chebyshev polynomials
    - Solutions to the hypergeometric differential equation
    - Many probability distributions (beta, F, Student's t)
    - Conformal mapping and complex analysis

    Algorithm
    ---------
    For ``|z| < 1``, uses direct series expansion with incremental term
    computation for numerical stability. Convergence is detected when term
    magnitude falls below machine epsilon.

    For ``|z| >= 1``, uses the linear transformation formula (DLMF 15.8.2):

    .. math::

       {}_2F_1(a,b;c;z) = G_1 (-z)^{-a} {}_2F_1(a, a-c+1; a-b+1; 1/z)
                        + G_2 (-z)^{-b} {}_2F_1(b, b-c+1; b-a+1; 1/z)

    where :math:`G_1` and :math:`G_2` are gamma function ratios. When a-b is
    an integer, a regularization technique is used to handle the pole.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Complex dtypes (complex64, complex128) are fully supported

    Autograd Support
    ----------------
    First-order derivatives are computed:
    - dF/dz = (a*b/c) * 2F1(a+1, b+1; c+1; z) (analytical)
    - dF/da, dF/db, dF/dc use finite differences

    Second-order derivatives are also supported via the backward_backward
    implementation.

    For complex inputs, Wirtinger derivative conventions are used:
    - First backward: grad_x = grad_output * conj(dF/dx)
    - Double backward follows the same convention

    Parameters
    ----------
    a : Tensor
        First numerator parameter. Broadcasting with b, c, and z is supported.
    b : Tensor
        Second numerator parameter. Broadcasting with a, c, and z is supported.
    c : Tensor
        Denominator parameter. Must not be a non-positive integer.
        Broadcasting with a, b, and z is supported.
    z : Tensor
        Input value. For real dtypes, convergence is best for ``|z| < 1``.
        Complex dtypes support all z via analytic continuation.
        Broadcasting with a, b, and c is supported.

    Returns
    -------
    Tensor
        The Gauss hypergeometric function 2F1(a, b; c; z) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([2.0])
    >>> c = torch.tensor([3.0])
    >>> z = torch.tensor([0.5])
    >>> hypergeometric_2_f_1(a, b, c, z)
    tensor([1.5452])

    Special case 2F1(a, b; c; 0) = 1:

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> c = torch.tensor([4.0])
    >>> z = torch.tensor([0.0])
    >>> hypergeometric_2_f_1(a, b, c, z)
    tensor([1.])

    Relation to (1-z)^(-a) when c = b:

    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> z = torch.tensor([0.5])
    >>> result = hypergeometric_2_f_1(a, b, b, z)
    >>> expected = (1 - z) ** (-a)
    >>> torch.allclose(result, expected)
    True

    Autograd:

    >>> z = torch.tensor([0.3], requires_grad=True)
    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([2.0])
    >>> c = torch.tensor([3.0])
    >>> y = hypergeometric_2_f_1(a, b, c, z)
    >>> y.backward()
    >>> z.grad  # Gradient w.r.t. z
    tensor([0.8403])

    .. warning:: Real z > 1 Limitation

       For real-valued inputs with z > 1 and non-integer (a - b), the
       mathematically correct result is generally complex. This implementation
       computes the full complex result internally but returns only the real
       part, which may be incorrect. To get accurate results for z > 1,
       use complex dtypes::

           >>> z_real = torch.tensor([2.0])  # May give incorrect real part
           >>> z_complex = torch.tensor([2.0 + 0j])  # Gives correct complex result

    .. warning:: Poles at non-positive integer c

       For c close to a non-positive integer, the result may be NaN.

    .. warning:: Slow convergence near unit circle

       For ``|z|`` close to 1, convergence may be slow.

    .. warning:: Reduced precision for large ``|z|``

       For ``|z| > 1`` with near-integer (a - b), numerical precision may be
       reduced.

    .. warning:: Overflow for large parameters

       For very large parameters, numerical overflow may occur.

    .. warning:: Finite difference gradients for parameters

       Parameter gradients (a, b, c) use finite differences and may have
       reduced accuracy compared to the analytical z gradient.

    Notes
    -----
    - The linear transformation for ``|z| >= 1`` involves gamma function
      ratios that can overflow for large parameters. The implementation
      uses log-space computation where possible to mitigate this.
    - When a - b is close to an integer, the regularization technique
      interpolates between nearby non-integer values to maintain continuity.

    See Also
    --------
    incomplete_beta : Regularized incomplete beta function (uses 2F1 internally)
    gamma : Gamma function
    """
    return torch.ops.torchscience.hypergeometric_2_f_1(a, b, c, z)
