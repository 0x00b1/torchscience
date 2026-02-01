import torch
from torch import Tensor


def dawson(z: Tensor) -> Tensor:
    r"""
    Dawson's integral (Dawson function).

    Computes Dawson's integral D(z) evaluated at each element of the input
    tensor.

    Mathematical Definition
    -----------------------
    Dawson's integral is defined as:

    .. math::

       D(z) = e^{-z^2} \int_0^z e^{t^2} \, dt

    This is related to the Faddeeva function w(z) by:

    - For real x: D(x) = sqrt(pi)/2 * Im[w(x)]
    - For complex z: D(z) = sqrt(pi)/2 * i * (exp(-z^2) - w(z))

    Special Values
    --------------
    - D(0) = 0
    - D(x) is odd: D(-x) = -D(x)
    - D(x) has a maximum at x ~ 0.924 where D(x) ~ 0.541
    - D(x) -> 1/(2x) as x -> infinity (asymptotic)

    Domain
    ------
    - z: any real or complex number
    - The function is entire (analytic everywhere in the complex plane)
    - No poles or branch cuts

    Algorithm
    ---------
    Uses the Faddeeva function w(z) for numerical evaluation:
    - For real x: D(x) = sqrt(pi)/2 * Im[w(x)]
    - For complex z: D(z) = sqrt(pi)/2 * i * (exp(-z^2) - w(z))

    This provides high accuracy across the entire domain.

    Applications
    ------------
    Dawson's integral appears in:

    - **Plasma physics**: The plasma dispersion function Z(z) = i*sqrt(pi)*w(z)
      is related to Dawson's function
    - **Spectroscopy**: Related to the Voigt profile through the Faddeeva
      function
    - **Heat conduction**: Appears in solutions of heat conduction problems
    - **Probability theory**: Related to the error function and Mills ratio

    Related Functions
    -----------------
    - Faddeeva function: w(x) = exp(-x^2) + 2i/sqrt(pi) * D(x) for real x
    - Error function: erf(x) = 2x/sqrt(pi) * sum_{n=0}^{inf} (-1)^n * D_n(x)
    - Voigt profile: Uses Dawson's function in its computation

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - For real inputs, output is real
    - For complex inputs, output is complex
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dz} D(z) = 1 - 2z \cdot D(z)

    Second-order derivatives (gradgradcheck) are also supported, computed
    using:

    .. math::

       \frac{d^2}{dz^2} D(z) = (4z^2 - 2) D(z) - 2z

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        Dawson's integral D(z) evaluated at each element of z.
        Output dtype matches input dtype (real for real, complex for complex).

    Examples
    --------
    Basic evaluation at real arguments:

    >>> z = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    >>> dawson(z)
    tensor([0.0000, 0.4244, 0.5381, 0.3013])

    At the origin:

    >>> z = torch.tensor([0.0], dtype=torch.float64)
    >>> dawson(z)
    tensor([0.])

    Maximum value (around x = 0.924):

    >>> z = torch.tensor([0.924], dtype=torch.float64)
    >>> dawson(z)
    tensor([0.5410])

    Odd function property:

    >>> z = torch.tensor([1.0], dtype=torch.float64)
    >>> dawson(z)
    tensor([0.5381])
    >>> dawson(-z)
    tensor([-0.5381])

    Autograd:

    >>> z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    >>> y = dawson(z)
    >>> y.backward()
    >>> z.grad  # Should be 1 - 2*z*D(z) = 1 - 2*1.0*0.5381 ~ -0.0762
    tensor([-0.0762])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j], dtype=torch.complex128)
    >>> dawson(z)
    tensor([0.6060-0.0985j])

    Notes
    -----
    - The function is numerically stable across the entire domain
    - For very large |z|, the asymptotic form D(z) ~ 1/(2z) is used
    - The implementation uses the Faddeeva function internally, which
      provides full complex plane support

    See Also
    --------
    scipy.special.dawsn : SciPy's implementation of Dawson's integral
    faddeeva_w : The Faddeeva function used in this implementation
    """
    return torch.ops.torchscience.dawson(z)
