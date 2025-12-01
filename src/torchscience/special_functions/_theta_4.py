import torch
from torch import Tensor


def theta_4(z: Tensor, q: Tensor) -> Tensor:
    r"""
    Jacobi theta function theta_4(z, q).

    Computes the Jacobi theta function of the fourth kind, defined as:

    .. math::

        \theta_4(z, q) = 1 + 2 \sum_{n=1}^{\infty} (-1)^n q^{n^2} \cos(2nz)

    This is an even function of z that is quasi-periodic with period pi.

    Parameters
    ----------
    z : Tensor
        The argument. Can be real or complex.
    q : Tensor
        The nome. Must satisfy |q| < 1 for convergence.

    Returns
    -------
    Tensor
        The value of theta_4(z, q).

    Examples
    --------
    >>> z = torch.tensor([0.0])
    >>> q = torch.tensor([0.0])
    >>> theta_4(z, q)
    tensor([1.])  # theta_4(z, 0) = 1

    >>> z = torch.tensor([0.0])
    >>> q = torch.tensor([0.5])
    >>> theta_4(z, q)
    tensor([0.6211])

    Notes
    -----
    - theta_4(z, 0) = 1 for all z
    - theta_4(-z, q) = theta_4(z, q) (even function)
    - theta_4 is related to the Jacobi elliptic functions
    - The nome q is related to the elliptic modulus by q = exp(-pi*K'/K)
    """
    return torch.ops.torchscience.theta_4(z, q)
