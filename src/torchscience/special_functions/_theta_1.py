import torch
from torch import Tensor


def theta_1(z: Tensor, q: Tensor) -> Tensor:
    r"""
    Jacobi theta function theta_1(z, q).

    Computes the Jacobi theta function of the first kind, defined as:

    .. math::

        \theta_1(z, q) = 2 \sum_{n=0}^{\infty} (-1)^n q^{(n+1/2)^2} \sin((2n+1)z)

    This is an odd function of z that is quasi-periodic with period pi.

    Parameters
    ----------
    z : Tensor
        The argument. Can be real or complex.
    q : Tensor
        The nome. Must satisfy |q| < 1 for convergence.

    Returns
    -------
    Tensor
        The value of theta_1(z, q).

    Examples
    --------
    >>> z = torch.tensor([0.0])
    >>> q = torch.tensor([0.5])
    >>> theta_1(z, q)
    tensor([0.])  # theta_1(0, q) = 0 (odd function)

    >>> z = torch.tensor([1.0])
    >>> q = torch.tensor([0.1])
    >>> theta_1(z, q)
    tensor([0.5894])

    Notes
    -----
    - theta_1(0, q) = 0 for all q (odd function)
    - theta_1(-z, q) = -theta_1(z, q)
    - theta_1 is related to the Jacobi elliptic functions
    - The nome q is related to the elliptic modulus by q = exp(-pi*K'/K)
      where K and K' are complete elliptic integrals
    """
    return torch.ops.torchscience.theta_1(z, q)
