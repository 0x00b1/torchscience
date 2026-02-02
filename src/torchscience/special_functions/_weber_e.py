import torch
from torch import Tensor


def weber_e(n: Tensor, z: Tensor) -> Tensor:
    r"""
    Weber function E_nu(z).

    Computes the Weber function E_nu(z) evaluated at each element of the
    input tensors, where n (nu) is the order and z is the argument.

    Mathematical Definition
    -----------------------
    The Weber function is defined by the integral:

    .. math::

       \mathbf{E}_\nu(z) = \frac{1}{\pi} \int_0^\pi
           \sin(\nu\theta - z\sin\theta) \, d\theta

    Relation to Bessel Functions
    ----------------------------
    For non-integer orders, the Weber function can be expressed as:

    .. math::

       \mathbf{E}_\nu(z) = -Y_\nu(z) + \frac{1}{\pi}
           \left[(\cos(\nu\pi)-1) A_\nu(z) - (\cos(\nu\pi)+1) A_{-\nu}(z)\right]

    where Y_nu is the Bessel function of the second kind.

    Relation to Anger Function
    --------------------------
    The Weber function E_nu and Anger function J_nu are paired solutions
    arising from the same integral representation but with sine and cosine:

    - E_nu uses sin in the integrand
    - J_nu uses cos in the integrand

    Note that contrary to some simplified references, the derivative
    d/dnu E_nu(z) is NOT equal to -J_nu(z). The actual relationship involves
    an additional factor of theta in the integrand.

    Derivative Formulas
    -------------------
    The derivative with respect to z follows a recurrence relation:

    .. math::

       \frac{\partial}{\partial z} \mathbf{E}_\nu(z) =
           \frac{1}{2}\left[\mathbf{E}_{\nu-1}(z) - \mathbf{E}_{\nu+1}(z)\right]

    The derivative with respect to order nu involves a modified integral:

    .. math::

       \frac{\partial}{\partial\nu} \mathbf{E}_\nu(z) =
           \frac{1}{\pi} \int_0^\pi \theta \cos(\nu\theta - z\sin\theta) \, d\theta

    Domain
    ------
    - n (nu): any real number (order)
    - z: any real number (argument)

    Applications
    ------------
    The Weber function appears in:
    - Solutions to certain differential equations
    - Wave propagation problems
    - Mathematical physics applications involving generalized Bessel-type functions
    - Often paired with the Anger function J_nu(z)

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
        Order of the Weber function. Can be any real number.
        Broadcasting with z is supported.
    z : Tensor
        Argument at which to evaluate the Weber function.
        Can be any real number.
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        The Weber function E_nu(z) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> n = torch.tensor([0.0, 1.0, 2.0])
    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> weber_e(n, z)

    Non-integer orders:

    >>> n = torch.tensor([0.5, 1.5])
    >>> z = torch.tensor([1.0, 2.0])
    >>> weber_e(n, z)

    Broadcasting:

    >>> n = torch.tensor([[0.0], [1.0]])  # (2, 1)
    >>> z = torch.tensor([1.0, 2.0, 3.0])  # (3,)
    >>> weber_e(n, z).shape
    torch.Size([2, 3])

    Autograd:

    >>> n = torch.tensor([0.5])
    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = weber_e(n, z)
    >>> y.backward()
    >>> z.grad is not None
    True

    See Also
    --------
    anger_j : Anger function J_nu(z)
    bessel_y : Bessel function of the second kind Y_nu(z)
    struve_h : Struve function H_nu(z)
    """
    return torch.ops.torchscience.weber_e(n, z)
