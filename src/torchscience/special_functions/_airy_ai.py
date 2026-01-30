import torch
from torch import Tensor


def airy_ai(x: Tensor) -> Tensor:
    r"""
    Airy function of the first kind.

    Computes the Airy function Ai(x) evaluated at each element of the
    input tensor.

    Mathematical Definition
    -----------------------
    The Airy function of the first kind is defined as:

    .. math::

       \text{Ai}(x) = \frac{1}{\pi} \int_0^\infty \cos\left(\frac{t^3}{3} + xt\right) \, dt

    It is one of two linearly independent solutions to the Airy differential
    equation:

    .. math::

       y'' - xy = 0

    Special Values
    --------------
    - Ai(0) = 1 / (3^{2/3} * Gamma(2/3)) ~ 0.35503
    - Ai'(0) = -1 / (3^{1/3} * Gamma(1/3)) ~ -0.25882
    - Ai(+inf) = 0 (exponential decay)
    - Ai(-inf) = 0 (oscillates with decreasing amplitude)
    - Ai(NaN) = NaN

    Asymptotic Behavior
    -------------------
    For large positive x:

    .. math::

       \text{Ai}(x) \sim \frac{\exp(-\frac{2}{3}x^{3/2})}{2\sqrt{\pi} \, x^{1/4}}

    For large negative x (oscillatory):

    .. math::

       \text{Ai}(-x) \sim \frac{\sin(\frac{2}{3}x^{3/2} + \frac{\pi}{4})}{\sqrt{\pi} \, x^{1/4}}

    Domain
    ------
    - x: any real or complex value
    - Ai(x) is an entire function (no singularities or branch cuts)

    Algorithm
    ---------
    - For |x| < 0.25: Taylor series around x=0
    - For positive x: Asymptotic expansion or polynomial approximation
    - For negative x: Asymptotic oscillatory expansion or series
    - Complex inputs use Taylor series convergent for all z

    Applications
    ------------
    The Airy function appears in many physical contexts:

    - Quantum mechanics: WKB approximation turning points
    - Optics: Diffraction near caustics
    - Electromagnetism: Radio wave propagation
    - Fluid dynamics: Free surface flows
    - Probability: Tracy-Widom distribution (random matrix theory)

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

       \frac{d}{dx} \text{Ai}(x) = \text{Ai}'(x)

    where Ai'(x) is the derivative of the Airy function.

    Second-order derivatives use the Airy differential equation:

    .. math::

       \frac{d^2}{dx^2} \text{Ai}(x) = x \cdot \text{Ai}(x)

    Parameters
    ----------
    x : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Airy function Ai evaluated at each element of x.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
    >>> airy_ai(x)
    tensor([ 0.3550,  0.1353,  0.0349,  0.5356,  0.2274])

    Special values:

    >>> airy_ai(torch.tensor(0.0))
    tensor(0.3550)

    >>> airy_ai(torch.tensor(5.0))  # Exponential decay for large positive x
    tensor(0.0011)

    >>> airy_ai(torch.tensor(-5.0))  # Oscillatory for negative x
    tensor(0.3508)

    Autograd:

    >>> x = torch.tensor([1.0], requires_grad=True)
    >>> y = airy_ai(x)
    >>> y.backward()
    >>> x.grad  # equals Ai'(1.0)
    tensor([-0.1591])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> airy_ai(z)
    tensor([0.1164-0.0755j, 0.0269+0.0282j])

    Verify the Airy differential equation y'' = x*y:

    >>> x = torch.tensor([2.0], requires_grad=True)
    >>> y = airy_ai(x)
    >>> # First derivative
    >>> (grad_y,) = torch.autograd.grad(y, x, create_graph=True)
    >>> # Second derivative
    >>> (grad2_y,) = torch.autograd.grad(grad_y, x)
    >>> # Check: y'' should equal x * y
    >>> torch.allclose(grad2_y, x.detach() * y.detach(), rtol=1e-4)
    True

    Zeros of Ai:

    >>> # First few zeros: ~-2.338, ~-4.088, ~-5.521
    >>> x = torch.tensor([-2.338])
    >>> airy_ai(x).abs() < 1e-3
    tensor([True])

    Notes
    -----
    - For purely real inputs with |x| large, the asymptotic expansions
      provide high accuracy.
    - Complex inputs use Taylor series which converges for all finite z,
      but may be slower for large |z|.
    - The Airy function is related to Bessel functions of order 1/3.

    See Also
    --------
    airy_bi : Airy function of the second kind Bi(x)
    """
    return torch.ops.torchscience.airy_ai(x)
