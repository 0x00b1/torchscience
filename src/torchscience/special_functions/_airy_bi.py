import torch
from torch import Tensor


def airy_bi(x: Tensor) -> Tensor:
    r"""
    Airy function of the second kind.

    Computes the Airy function Bi(x) evaluated at each element of the
    input tensor.

    Mathematical Definition
    -----------------------
    The Airy function of the second kind is defined as:

    .. math::

       \text{Bi}(x) = \frac{1}{\pi} \int_0^\infty \left[\exp\left(-\frac{t^3}{3} + xt\right) + \sin\left(\frac{t^3}{3} + xt\right)\right] \, dt

    It is the second linearly independent solution to the Airy differential
    equation:

    .. math::

       y'' - xy = 0

    Special Values
    --------------
    - Bi(0) = 1 / (3^{1/6} * Gamma(2/3)) ~ 0.61493
    - Bi'(0) = 3^{1/6} / Gamma(1/3) ~ 0.44829
    - Bi(+inf) = +inf (exponential growth)
    - Bi(-inf) = 0 (oscillates with decreasing amplitude)
    - Bi(NaN) = NaN

    Asymptotic Behavior
    -------------------
    For large positive x (exponential growth):

    .. math::

       \text{Bi}(x) \sim \frac{\exp(\frac{2}{3}x^{3/2})}{\sqrt{\pi} \, x^{1/4}}

    For large negative x (oscillatory):

    .. math::

       \text{Bi}(-x) \sim \frac{\cos(\frac{2}{3}x^{3/2} + \frac{\pi}{4})}{\sqrt{\pi} \, x^{1/4}}

    Domain
    ------
    - x: any real or complex value
    - Bi(x) is an entire function (no singularities or branch cuts)

    Algorithm
    ---------
    - For |x| < 2.09: Taylor series around x=0
    - For positive x: Asymptotic expansion with exponential growth
    - For negative x: Asymptotic oscillatory expansion
    - Complex inputs use Taylor series convergent for all z

    Applications
    ------------
    The Airy function Bi appears in many physical contexts:

    - Quantum mechanics: WKB approximation turning points (along with Ai)
    - Optics: Diffraction patterns near caustics
    - Electromagnetism: Radio wave propagation
    - Fluid dynamics: Free surface flows

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when x.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dx} \text{Bi}(x) = \text{Bi}'(x)

    where Bi'(x) is the derivative of the Airy function.

    Second-order derivatives use the Airy differential equation:

    .. math::

       \frac{d^2}{dx^2} \text{Bi}(x) = x \cdot \text{Bi}(x)

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Airy function Bi evaluated at each element of x.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
    >>> airy_bi(x)
    tensor([ 0.6149,  1.2074,  3.2981,  0.1039,  0.0412])

    Special values:

    >>> airy_bi(torch.tensor(0.0))
    tensor(0.6149)

    >>> airy_bi(torch.tensor(5.0))  # Exponential growth for large positive x
    tensor(657.7922)

    >>> airy_bi(torch.tensor(-5.0))  # Oscillatory for negative x
    tensor(-0.1383)

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = airy_bi(x)
    >>> y.backward()
    >>> x.grad  # equals Bi'(1.0)
    tensor([0.9324])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> airy_bi(z)
    tensor([1.0886+0.5634j, 3.0259-1.5069j])

    Verify the Airy differential equation y'' = x*y:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = airy_bi(x)
    >>> # First derivative
    >>> (grad_y,) = torch.autograd.grad(y, x, create_graph=True)
    >>> # Second derivative
    >>> (grad2_y,) = torch.autograd.grad(grad_y, x)
    >>> # Check: y'' should equal x * y
    >>> torch.allclose(grad2_y, x.detach() * y.detach(), rtol=1e-4)
    True

    Zeros of Bi:

    >>> # First few zeros: ~-1.174, ~-3.271, ~-4.831
    >>> x = torch.tensor([-1.174])
    >>> airy_bi(x).abs() < 1e-2
    tensor([True])

    Notes
    -----
    - For purely real inputs with |x| large, the asymptotic expansions
      provide high accuracy.
    - Complex inputs use Taylor series which converges for all finite z,
      but may be slower for large |z|.
    - The Airy function Bi is related to Bessel functions of order 1/3.
    - For large positive x, Bi(x) grows exponentially and may overflow.

    See Also
    --------
    airy_ai : Airy function of the first kind Ai(x)
    """
    return torch.ops.torchscience.airy_bi(x)
