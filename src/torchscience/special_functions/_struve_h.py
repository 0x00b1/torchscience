import torch
from torch import Tensor


def struve_h(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Struve function of the first kind of general order n.

    Computes the Struve function H_n(z) evaluated at each element of the
    input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The Struve function of the first kind of order n is defined as:

    .. math::

       \mathbf{H}_n(z) = \sum_{k=0}^\infty \frac{(-1)^k}
           {\Gamma(k+\tfrac{3}{2})\Gamma(k+n+\tfrac{3}{2})}
           \left(\frac{z}{2}\right)^{n+2k+1}

    Or equivalently:

    .. math::

       \mathbf{H}_n(z) = \left(\frac{z}{2}\right)^{n+1}
           \sum_{k=0}^\infty \frac{(-1)^k (z/2)^{2k}}
           {\Gamma(k+\tfrac{3}{2})\Gamma(k+n+\tfrac{3}{2})}

    Special Values
    --------------
    - H_n(0) = 0 for all n >= -1
    - H_0(z) matches struve_h_0(z)
    - H_1(z) matches struve_h_1(z)

    Symmetry
    --------
    For integer n:

    .. math::

       \mathbf{H}_n(-z) = (-1)^{n+1} \mathbf{H}_n(z)

    Domain
    ------
    - n: any real or complex number (order)
    - z: any real or complex number (argument)
    - For n < -1 at z=0, the function may be singular

    Algorithm
    ---------
    - Uses power series expansion which converges for all z
    - For n = 0 or n = 1, specialized implementations may be faster
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Derivative Formulas
    -------------------
    The derivative with respect to z is:

    .. math::

       \frac{\partial}{\partial z} \mathbf{H}_n(z) =
           \frac{1}{2}\left[\mathbf{H}_{n-1}(z) - \mathbf{H}_{n+1}(z)\right]
           + \frac{(z/2)^n}{\sqrt{\pi}\Gamma(n+\tfrac{3}{2})}

    Applications
    ------------
    The Struve function appears in many contexts:
    - Electromagnetic wave propagation
    - Diffraction theory
    - Fluid dynamics
    - Heat conduction problems
    - Related to Bessel functions in integral representations

    Relation to Modified Struve Function
    -------------------------------------
    The modified Struve function L_n(z) is related by:

    .. math::

       \mathbf{L}_n(z) = -i e^{-i n \pi/2} \mathbf{H}_n(iz)

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported for both n and z when they require grad.
    The gradient with respect to z is computed analytically using the
    recurrence relation. The gradient with respect to n is computed
    numerically since the analytical formula is complex.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the Struve function. Can be any real or complex number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the Struve function.
        Can be any real or complex number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Struve function H_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> struve_h(n, z)
    tensor([0.5683, 0.6461, 0.7254])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(struve_h(n0, z), struve_h_0(z))
    True

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> struve_h(n, z)
    tensor([0.4244, 0.5671, 0.6806])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> struve_h(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = struve_h(n, z)
    >>> y.backward()
    >>> z.grad is not None
    True

    Symmetry for integer n:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> h_pos = struve_h(n, z)
    >>> h_neg = struve_h(n, -z)
    >>> torch.allclose(h_neg, -h_pos)  # H_2(-z) = -H_2(z)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> struve_h(n, z)
    tensor([0.5476+0.3068j])

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision due to
       the oscillatory nature of Struve functions and potential numerical
       cancellation.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `struve_h_0` and
      `struve_h_1` may provide slightly better accuracy.
    - The power series converges for all z, but may be slow for large |z|.

    See Also
    --------
    struve_h_0 : Struve function of order zero
    struve_h_1 : Struve function of order one
    struve_l : Modified Struve function of general order
    struve_l_0 : Modified Struve function of order zero
    struve_l_1 : Modified Struve function of order one
    """
    return torch.ops.torchscience.struve_h(n, z)
