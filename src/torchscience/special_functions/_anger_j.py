import torch
from torch import Tensor


def anger_j(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Anger function J_nu(z).

    Computes the Anger function J_nu(z) evaluated at each element of the
    input tensors, where n (nu) is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The Anger function is defined by the integral:

    .. math::

       \mathbf{J}_\nu(z) = \frac{1}{\pi} \int_0^\pi
           \cos(\nu\theta - z\sin\theta) \, d\theta

    Special Cases
    -------------
    For integer orders n, the Anger function equals the Bessel function
    of the first kind:

    .. math::

       \mathbf{J}_n(z) = J_n(z) \quad \text{for integer } n

    For non-integer orders, there's a correction term involving the
    auxiliary function A_nu(z):

    .. math::

       \mathbf{J}_\nu(z) = J_\nu(z) + \frac{\sin(\nu\pi)}{\pi}
           \left[A_\nu(z) + A_{-\nu}(z)\right]

    Derivative Formulas
    -------------------
    The derivative with respect to z follows a recurrence relation:

    .. math::

       \frac{\partial}{\partial z} \mathbf{J}_\nu(z) =
           \frac{1}{2}\left[\mathbf{J}_{\nu-1}(z) - \mathbf{J}_{\nu+1}(z)\right]

    The derivative with respect to order nu involves a modified integral:

    .. math::

       \frac{\partial}{\partial\nu} \mathbf{J}_\nu(z) =
           -\frac{1}{\pi} \int_0^\pi \theta \sin(\nu\theta - z\sin\theta) \, d\theta

    Relation to Weber Function
    --------------------------
    The Anger function J_nu and Weber function E_nu are paired solutions
    arising from the same integral representation but with cosine and sine:

    - J_nu uses cos in the integrand
    - E_nu uses sin in the integrand

    Note that contrary to some simplified references, the derivative
    d/dnu J_nu(z) is NOT equal to E_nu(z). The actual relationship involves
    an additional factor of theta in the integrand.

    Domain
    ------
    - n (nu): any real number (order)
    - z: any real number (argument)

    Applications
    ------------
    The Anger function appears in:
    - Solutions to certain differential equations
    - Wave propagation problems
    - Mathematical physics applications involving generalized Bessel-type functions

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float32, float64

    Autograd Support
    ----------------
    Gradients are fully supported for both n and z when they require grad.
    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Order of the Anger function. Can be any real number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the Anger function.
        Can be any real number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Anger function J_nu(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage with integer orders (equals Bessel J):

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> anger_j(n, z)
    tensor([0.7652, 0.5767, 0.4861])

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5])
    >>> z = torch.tensor([1.0, 2.0])
    >>> anger_j(n, z)

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0]])  # (2, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])  # (3,)
    >>> anger_j(n, z).shape
    torch.Size([2, 3])

    Autograd:

    >>> n = torch.tensor([0.5])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = anger_j(n, z)
    >>> y.backward()
    >>> z.grad is not None
    True

    See Also
    --------
    weber_e : Weber function E_nu(z)
    bessel_j : Bessel function of the first kind J_nu(z)
    struve_h : Struve function H_nu(z)
    """
    return torch.ops.torchscience.anger_j(n, z)
