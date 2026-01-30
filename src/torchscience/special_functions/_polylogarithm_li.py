import torch
from torch import Tensor


def polylogarithm_li(s: Tensor, z: Tensor) -> Tensor:
    r"""
    Polylogarithm function.

    Computes the polylogarithm function Li_s(z) evaluated element-wise.

    Mathematical Definition
    -----------------------
    The polylogarithm is defined for |z| <= 1 as:

    .. math::

        \text{Li}_s(z) = \sum_{k=1}^{\infty} \frac{z^k}{k^s}

    This infinite series converges for |z| < 1 for all s, and converges for
    |z| = 1 when Re(s) > 1.

    Special Cases
    -------------
    - Li_0(z) = z / (1 - z)
    - Li_1(z) = -ln(1 - z)
    - Li_2(z) is the dilogarithm (Spence's function)
    - Li_3(z) is the trilogarithm
    - Li_s(0) = 0
    - Li_s(1) = zeta(s) for Re(s) > 1 (Riemann zeta function)
    - Li_2(1) = pi^2 / 6

    Special Values
    --------------
    - Li_2(0.5) = pi^2/12 - (ln(2))^2/2 approximately equals 0.5822
    - Li_2(1) = pi^2/6 approximately equals 1.6449
    - Li_2(-1) = -pi^2/12 approximately equals -0.8225
    - Li_3(0.5) approximately equals 0.5372
    - Li_n(1) = zeta(n) for n > 1

    Domain Restrictions
    -------------------
    - |z| <= 1: Valid domain for this implementation
    - |z| > 1: Returns NaN (analytic continuation not implemented)
    - z = 1: Equals zeta(s) for s > 1, infinity for s <= 1

    Algorithm
    ---------
    Uses direct series summation with convergence acceleration for |z| close to 1.
    The series is:

    .. math::

        \text{Li}_s(z) = z + \frac{z^2}{2^s} + \frac{z^3}{3^s} + \cdots

    For |z| < 0.5, the series converges rapidly. For |z| closer to 1, more
    terms are used to ensure accuracy.

    Applications
    ------------
    The polylogarithm appears in:
    - Quantum electrodynamics calculations
    - Statistical mechanics (Fermi-Dirac and Bose-Einstein statistics)
    - Number theory (relations with the Riemann zeta function)
    - Algebraic K-theory
    - Cluster algebra
    - Goncharov's conjecture on periods

    Dtype Promotion
    ---------------
    - Both inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported for both parameters.

    Derivative with respect to z:

    .. math::

        \frac{\partial}{\partial z} \text{Li}_s(z) = \frac{\text{Li}_{s-1}(z)}{z}

    Derivative with respect to s:

    .. math::

        \frac{\partial}{\partial s} \text{Li}_s(z) = -\sum_{k=1}^{\infty} \frac{\ln(k) \cdot z^k}{k^s}

    Second-order derivatives are also supported.

    Parameters
    ----------
    s : Tensor
        The order parameter. Can be any real or complex number.
        Broadcasting with z is supported.
    z : Tensor
        The argument. Must satisfy |z| <= 1 for real results. Can be real
        or complex. Broadcasting with s is supported.

    Returns
    -------
    Tensor
        The polylogarithm Li_s(z) evaluated at each element.
        Returns NaN for |z| > 1.

    Examples
    --------
    Evaluate the dilogarithm Li_2:

    >>> import torch
    >>> import math
    >>> s = torch.tensor([2.0])
    >>> z = torch.tensor([0.5])
    >>> polylogarithm_li(s, z)
    tensor([0.5822])

    Verify Li_2(1) = pi^2/6:

    >>> s = torch.tensor([2.0], dtype=torch.float64)
    >>> z = torch.tensor([1.0], dtype=torch.float64)
    >>> result = polylogarithm_li(s, z)
    >>> expected = math.pi**2 / 6
    >>> torch.isclose(result, torch.tensor([expected], dtype=torch.float64), rtol=1e-4)
    tensor([True])

    Li_1(z) = -ln(1-z):

    >>> s = torch.tensor([1.0], dtype=torch.float64)
    >>> z = torch.tensor([0.5], dtype=torch.float64)
    >>> result = polylogarithm_li(s, z)
    >>> expected = -torch.log(1 - z)
    >>> torch.isclose(result, expected, rtol=1e-6)
    tensor([True])

    Broadcasting:

    >>> s = torch.tensor([2.0, 3.0])
    >>> z = torch.tensor([0.5])
    >>> polylogarithm_li(s, z)
    tensor([0.5822, 0.5372])

    Autograd:

    >>> s = torch.tensor([2.0], requires_grad=True)
    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> y = polylogarithm_li(s, z)
    >>> y.backward()
    >>> z.grad  # derivative w.r.t. z
    tensor([1.3863])

    Notes
    -----
    - For |z| > 1, the function returns NaN. To extend to the full complex
      plane, one would need to use the inversion formula and handle branch cuts.
    - The dilogarithm Li_2 is related to Spence's function by:
      spence(z) = Li_2(1-z) in scipy convention, or Li_2(z) = -spence(1-z)
    - For integer s <= 0, the polylogarithm has closed forms in terms of
      rational functions.

    See Also
    --------
    zeta : Riemann zeta function (Li_s(1) for s > 1)
    scipy.special.spence : Spence's function (related to Li_2)
    """
    return torch.ops.torchscience.polylogarithm_li(s, z)
