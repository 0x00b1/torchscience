import torch
from torch import Tensor


def erfcinv(x: Tensor) -> Tensor:
    r"""
    Inverse complementary error function.

    Computes the inverse complementary error function evaluated at each element
    of the input tensor. The inverse complementary error function is defined as
    the function y = erfcinv(x) such that erfc(y) = x.

    Mathematical Definition
    -----------------------
    The inverse complementary error function is defined implicitly by:

    .. math::

       \text{erfc}(\text{erfcinv}(x)) = x

    where the complementary error function is:

    .. math::

       \text{erfc}(y) = 1 - \text{erf}(y) = \frac{2}{\sqrt{\pi}} \int_y^\infty e^{-t^2} \, dt

    Relationship to erfinv
    ----------------------
    The inverse complementary error function is related to erfinv by:

    .. math::

       \text{erfcinv}(x) = \text{erfinv}(1 - x)

    Domain and Range
    ----------------
    - Domain: x in (0, 2)
    - Range: all real numbers
    - erfcinv(0) = +inf
    - erfcinv(1) = 0
    - erfcinv(2) = -inf
    - erfcinv(x) is undefined (NaN) for x < 0 or x > 2

    Special Values
    --------------
    - erfcinv(0) = +inf
    - erfcinv(0.5) = 0.4769362762044699...
    - erfcinv(1) = 0
    - erfcinv(1.5) = -0.4769362762044699...
    - erfcinv(2) = -inf

    Algorithm
    ---------
    - For most values: uses erfcinv(x) = erfinv(1-x)
    - For x near 0: uses asymptotic expansion with Newton refinement
    - Newton-Raphson refinement for full precision

    Applications
    ------------
    The inverse complementary error function is useful in:
    - Statistics: computing quantiles of half-normal distributions
    - Tail probability analysis: working with extreme values
    - Quality control: defect rate calculations (six sigma)
    - Communications: bit error rate (BER) analysis
    - Reliability engineering: failure rate modeling

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

       \frac{d}{dx} \text{erfcinv}(x) = -\frac{\sqrt{\pi}}{2} \exp(\text{erfcinv}(x)^2)

    Note the negative sign compared to erfinv, since erfcinv is monotonically
    decreasing.

    Second-order derivatives (gradgradcheck) are also supported:

    .. math::

       \frac{d^2}{dx^2} \text{erfcinv}(x) = \frac{\pi}{2} \text{erfcinv}(x) \exp(2 \cdot \text{erfcinv}(x)^2)

    Parameters
    ----------
    x : Tensor
        Input tensor. Must have values in the range (0, 2) for finite output.

    Returns
    -------
    Tensor
        The inverse complementary error function evaluated at each element of x.
        Output dtype matches input dtype.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.5, 1.0, 1.5])
    >>> erfcinv(x)
    tensor([ 0.4769,  0.0000, -0.4769])

    Verify roundtrip with erfc:

    >>> x = torch.tensor([0.1, 1.0, 1.9])
    >>> torch.erfc(erfcinv(x))
    tensor([0.1000, 1.0000, 1.9000])

    Symmetry around x=1:

    >>> x = torch.tensor([0.1, 1.9])
    >>> erfcinv(x)  # erfcinv(0.1) = -erfcinv(1.9)
    tensor([ 1.1631, -1.1631])

    Values near the boundaries:

    >>> x = torch.tensor([0.01, 0.001, 0.0001])
    >>> erfcinv(x)
    tensor([1.8214, 2.3267, 2.7510])

    Autograd:

    >>> x = torch.tensor([0.5], requires_grad=True)
    >>> y = erfcinv(x)
    >>> y.backward()
    >>> x.grad  # -sqrt(pi)/2 * exp(erfcinv(0.5)^2) ~ -1.112
    tensor([-1.1124])

    .. warning:: Behavior at boundaries

       At x = 0, erfcinv returns +inf. At x = 2, it returns -inf.
       The derivative magnitude is also infinite at these points.
       For values very close to 0 or 2, the function values grow rapidly.

    See Also
    --------
    torch.erfc : Complementary error function (forward direction)
    erfinv : Inverse error function
    torch.special.erfcinv : PyTorch's built-in implementation
    """
    return torch.ops.torchscience.erfcinv(x)
