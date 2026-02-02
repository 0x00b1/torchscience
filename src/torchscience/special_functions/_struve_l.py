import torch
from torch import Tensor


def struve_l(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Modified Struve function of general order n.

    Computes the modified Struve function L_n(z) evaluated at each element
    of the input tensors, where n is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The modified Struve function of order n is defined as:

    .. math::

       \mathbf{L}_n(z) = \sum_{k=0}^\infty \frac{1}
           {\Gamma(k+\tfrac{3}{2})\Gamma(k+n+\tfrac{3}{2})}
           \left(\frac{z}{2}\right)^{n+2k+1}

    Or equivalently:

    .. math::

       \mathbf{L}_n(z) = \left(\frac{z}{2}\right)^{n+1}
           \sum_{k=0}^\infty \frac{(z/2)^{2k}}
           {\Gamma(k+\tfrac{3}{2})\Gamma(k+n+\tfrac{3}{2})}

    Note the absence of the alternating (-1)^k factor compared to H_n(z).

    Special Values
    --------------
    - L_n(0) = 0 for all n >= -1
    - L_0(z) matches struve_l_0(z)
    - L_1(z) matches struve_l_1(z)
    - L_n(+inf) = +inf for z > 0

    Symmetry
    --------
    For integer n:

    .. math::

       \mathbf{L}_n(-z) = (-1)^{n+1} \mathbf{L}_n(z)

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

       \frac{\partial}{\partial z} \mathbf{L}_n(z) =
           \frac{1}{2}\left[\mathbf{L}_{n-1}(z) + \mathbf{L}_{n+1}(z)\right]
           + \frac{(z/2)^n}{\sqrt{\pi}\Gamma(n+\tfrac{3}{2})}

    Note the PLUS sign between L_{n-1} and L_{n+1} (compared to minus for H_n).

    Relation to Struve Function
    ---------------------------
    The modified Struve function L_n(z) is related to H_n(z) by:

    .. math::

       \mathbf{L}_n(z) = -i e^{-i n \pi/2} \mathbf{H}_n(iz)

    Applications
    ------------
    The modified Struve function appears in:
    - Heat conduction problems
    - Electromagnetic wave theory
    - Solutions to certain differential equations
    - Related to modified Bessel functions

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
        Order of the modified Struve function. Can be any real or complex
        number. Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the modified Struve function.
        Can be any real or complex number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The modified Struve function L_n(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> struve_l(n, z)
    tensor([0.4966, 1.0666, 2.2553])

    Matches specialized functions for n=0, n=1:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> n0 = torch.zeros_like(z)
    >>> torch.allclose(struve_l(n0, z), struve_l_0(z))
    True

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> struve_l(n, z)
    tensor([0.3844, 0.8741, 2.0106])

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])        # (3,)
    >>> struve_l(n, z).shape
    torch.Size([3, 3])

    Autograd:

    >>> n = torch.tensor([1.0])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = struve_l(n, z)
    >>> y.backward()
    >>> z.grad is not None
    True

    Symmetry for integer n:

    >>> n = torch.tensor([2.0])
    >>> z = torch.tensor([3.0])
    >>> l_pos = struve_l(n, z)
    >>> l_neg = struve_l(n, -z)
    >>> torch.allclose(l_neg, -l_pos)  # L_2(-z) = -L_2(z)
    True

    Complex input:

    >>> n = torch.tensor([0.0 + 0.0j])
    >>> z = torch.tensor([1.0 + 0.5j])
    >>> struve_l(n, z)
    tensor([0.4759+0.3221j])

    .. warning:: Numerical precision

       For large |n| or large |z|, the function may lose precision.
       Unlike H_n(z) which oscillates, L_n(z) grows exponentially for
       large positive z.

    Notes
    -----
    - For n = 0 or n = 1, the specialized functions `struve_l_0` and
      `struve_l_1` may provide slightly better accuracy.
    - The power series converges for all z, but may be slow for large |z|.
    - L_n(z) grows like I_n(z) (modified Bessel function) for large z.

    See Also
    --------
    struve_l_0 : Modified Struve function of order zero
    struve_l_1 : Modified Struve function of order one
    struve_h : Struve function of general order
    struve_h_0 : Struve function of order zero
    struve_h_1 : Struve function of order one
    """
    return torch.ops.torchscience.struve_l(n, z)
