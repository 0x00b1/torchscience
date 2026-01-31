import torch
from torch import Tensor


def erfinv(x: Tensor) -> Tensor:
    r"""
    Inverse error function.

    Computes the inverse error function evaluated at each element of the input
    tensor. The inverse error function is defined as the function y = erfinv(x)
    such that erf(y) = x.

    Mathematical Definition
    -----------------------
    The inverse error function is defined implicitly by:

    .. math::

       \text{erf}(\text{erfinv}(x)) = x

    where the error function is:

    .. math::

       \text{erf}(y) = \frac{2}{\sqrt{\pi}} \int_0^y e^{-t^2} \, dt

    Domain and Range
    ----------------
    - Domain: x in (-1, 1)
    - Range: all real numbers
    - erfinv(-1) = -inf
    - erfinv(0) = 0
    - erfinv(1) = +inf
    - erfinv(x) is undefined (NaN) for |x| > 1

    Special Values
    --------------
    - erfinv(0) = 0
    - erfinv(0.5) = 0.4769362762044699...
    - erfinv(1) = +inf
    - erfinv(-1) = -inf

    Algorithm
    ---------
    - Uses Winitzki's rational approximation for initial estimate
    - Two Newton-Raphson refinement steps for full precision
    - Separate approximations for central (|x| < 0.9959) and tail regions

    Applications
    ------------
    The inverse error function is widely used in:
    - Statistics: converting between probability and z-scores
    - Gaussian quantiles: erfinv relates to the probit function
    - Random number generation: Box-Muller transform variants
    - Signal processing: noise analysis
    - Finance: option pricing and risk metrics

    Dtype Support
    -------------
    - Supports float16, bfloat16, float32, float64
    - Integer inputs are promoted to floating-point
    - Note: This is a real-only function (no complex support)

    Autograd Support
    ----------------
    Gradients are fully supported when x.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dx} \text{erfinv}(x) = \frac{\sqrt{\pi}}{2} \exp(\text{erfinv}(x)^2)

    Second-order derivatives (gradgradcheck) are also supported:

    .. math::

       \frac{d^2}{dx^2} \text{erfinv}(x) = \frac{\pi}{2} \text{erfinv}(x) \exp(2 \cdot \text{erfinv}(x)^2)

    Parameters
    ----------
    x : Tensor
        Input tensor. Must have values in the range (-1, 1) for finite output.

    Returns
    -------
    Tensor
        The inverse error function evaluated at each element of x.
        Output dtype matches input dtype.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 0.5, -0.5])
    >>> erfinv(x)
    tensor([ 0.0000,  0.4769, -0.4769])

    Verify roundtrip with erf:

    >>> x = torch.tensor([0.1, 0.5, 0.9])
    >>> torch.erf(erfinv(x))
    tensor([0.1000, 0.5000, 0.9000])

    Values near the boundaries:

    >>> x = torch.tensor([0.99, 0.999, 0.9999])
    >>> erfinv(x)
    tensor([1.8214, 2.3267, 2.7510])

    Autograd:

    >>> x = torch.tensor([0.5], requires_grad=True)
    >>> y = erfinv(x)
    >>> y.backward()
    >>> x.grad  # sqrt(pi)/2 * exp(erfinv(0.5)^2) ~ 1.112
    tensor([1.1124])

    .. warning:: Behavior at boundaries

       At x = +/-1, erfinv returns +/-inf. The derivative is also infinite
       at these points. For values very close to +/-1, the function values
       grow rapidly and may overflow in float32.

    See Also
    --------
    torch.erf : Error function (forward direction)
    erfcinv : Inverse complementary error function
    torch.special.erfinv : PyTorch's built-in implementation
    """
    return torch.ops.torchscience.erfinv(x)
