import torch
from torch import Tensor


def legendre_polynomial_q(x: Tensor, n: Tensor) -> Tensor:
    r"""
    Legendre function of the second kind.

    Computes the Legendre function of the second kind Q_n(x) for degree n
    and argument x.

    Mathematical Definition
    -----------------------
    For integer n >= 0 and |x| < 1, Q_n(x) is defined by the recurrence:

    .. math::

       Q_0(x) = \frac{1}{2} \ln\left(\frac{1+x}{1-x}\right) = \text{arctanh}(x)

       Q_1(x) = x Q_0(x) - 1

       (n+1) Q_{n+1}(x) = (2n+1) x Q_n(x) - n Q_{n-1}(x)

    Alternatively, Q_n can be expressed using the Legendre polynomial of the
    first kind P_n:

    .. math::

       Q_n(x) = P_n(x) \cdot \frac{1}{2} \ln\left(\frac{1+x}{1-x}\right) - W_{n-1}(x)

    where W_{n-1}(x) = sum_{k=1}^{n} (1/k) P_{k-1}(x) P_{n-k}(x).

    For non-integer n, the function is extended analytically using:

    .. math::

       Q_n(x) = \frac{\pi}{2} \cdot \frac{P_n(x)\cos(n\pi) - P_n(-x)}{\sin(n\pi)}

    Special Values
    --------------
    - Q_0(0) = 0
    - Q_1(0) = -1
    - Q_n(0) = 0 for even n
    - Q_n(0) non-zero for odd n
    - Q_n(x) has logarithmic singularities at x = +/- 1

    Domain
    ------
    - x: real values, typically |x| < 1 for convergence
    - n: any real value (integer values are the classical case)
    - At x = +/- 1: logarithmic singularity (returns +/- infinity)
    - For |x| > 1: results may be complex or involve different analytic continuation

    Algorithm
    ---------
    For integer n >= 0, uses the three-term recurrence relation starting from
    Q_0(x) = arctanh(x) and Q_1(x) = x*arctanh(x) - 1.

    For non-integer n, uses the formula involving P_n(x) and P_n(-x).

    Applications
    ------------
    The Legendre functions of the second kind appear in:
    - Electrostatics: potential theory for prolate spheroidal coordinates
    - Gravitational potential: expansion of 1/|r - r'| in Legendre functions
    - Quantum mechanics: scattering theory
    - Heat conduction: solutions in spherical coordinates with boundary conditions
    - Fluid dynamics: Stokes flow around ellipsoids

    Dtype Promotion
    ---------------
    - Standard PyTorch dtype promotion rules apply between x and n
    - Supports float16, bfloat16, float32, float64
    - Complex support is limited due to the logarithm in Q_0

    Integer Dtype Handling
    ----------------------
    If n is passed as an integer dtype tensor (e.g., torch.int32, torch.int64),
    it will be promoted to a floating-point dtype via PyTorch's standard type
    promotion rules before computation.

    Autograd Support
    ----------------
    - Gradients for x are computed analytically where possible
    - Gradients for n are computed via finite differences
    - Second-order derivatives are supported but may be approximate

    .. warning::

       The function has logarithmic singularities at x = +/- 1. Numerical
       results near these points may be inaccurate or infinite.

    .. warning::

       Second-order derivatives with respect to n are approximate and may
       return zero. Use with caution when computing Hessians involving the
       degree parameter n.

    Backward formulas:

    .. math::

       \frac{\partial Q_n}{\partial x} = \frac{n x Q_n(x) - n Q_{n-1}(x)}{x^2 - 1}

    For n = 0:

    .. math::

       \frac{\partial Q_0}{\partial x} = \frac{1}{1 - x^2}

    Parameters
    ----------
    x : Tensor
        Input tensor. Typically |x| < 1 for well-defined real results.
        Broadcasting with n is supported.
    n : Tensor
        Degree of the function. Can be integer or non-integer.
        For the classical Legendre functions, use non-negative integers.

    Returns
    -------
    Tensor
        The Legendre function Q_n(x) evaluated at the input values.
        Output dtype follows PyTorch promotion rules.

    Examples
    --------
    Basic computation for integer degrees:

    >>> x = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64)
    >>> n = torch.tensor([0.0], dtype=torch.float64)
    >>> legendre_polynomial_q(x, n)  # Q_0(x) = arctanh(x)
    tensor([ 0.0000,  0.5493, -0.5493], dtype=torch.float64)

    Higher degree:

    >>> x = torch.tensor([0.5], dtype=torch.float64)
    >>> n = torch.tensor([1.0], dtype=torch.float64)
    >>> legendre_polynomial_q(x, n)  # Q_1(x) = x*arctanh(x) - 1
    tensor([-0.7254], dtype=torch.float64)

    Verify Q_0(x) = arctanh(x):

    >>> x = torch.tensor([0.5], dtype=torch.float64)
    >>> n = torch.tensor([0.0], dtype=torch.float64)
    >>> legendre_polynomial_q(x, n)
    tensor([0.5493], dtype=torch.float64)
    >>> torch.arctanh(x)
    tensor([0.5493], dtype=torch.float64)

    Autograd example:

    >>> x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    >>> n = torch.tensor([0.0], dtype=torch.float64)
    >>> y = legendre_polynomial_q(x, n)
    >>> y.backward()
    >>> x.grad  # dQ_0/dx = 1/(1-x^2)
    tensor([1.3333], dtype=torch.float64)

    .. warning:: Singularities

       Q_n(x) has logarithmic singularities at x = +/- 1:

       >>> x = torch.tensor([0.9999], dtype=torch.float64)
       >>> n = torch.tensor([0.0], dtype=torch.float64)
       >>> legendre_polynomial_q(x, n)  # Large value, approaching infinity
       tensor([4.9517], dtype=torch.float64)

    Notes
    -----
    - Q_n(x) and P_n(x) are the two linearly independent solutions of
      Legendre's differential equation: (1-x^2)y'' - 2xy' + n(n+1)y = 0
    - For |x| > 1, the function may require complex extension
    - The Wronskian of P_n and Q_n satisfies: W[P_n, Q_n] = 1/(1-x^2)

    See Also
    --------
    scipy.special.lqn : SciPy's Legendre Q function
    legendre_polynomial_p : Legendre polynomial of the first kind P_n(x)
    """
    return torch.ops.torchscience.legendre_polynomial_q(x, n)
