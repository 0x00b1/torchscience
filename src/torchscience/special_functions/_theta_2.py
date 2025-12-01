import torch
from torch import Tensor


def theta_2(z: Tensor, q: Tensor) -> Tensor:
    r"""
    Jacobi theta function theta_2(z, q).

    Computes the Jacobi theta function of the second kind, defined as:

    .. math::

        \theta_2(z, q) = 2 \sum_{n=0}^{\infty} q^{(n+1/2)^2} \cos((2n+1)z)

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
        The value of theta_2(z, q).

    Examples
    --------
    >>> z = torch.tensor([0.0])
    >>> q = torch.tensor([0.5])
    >>> theta_2(z, q)
    tensor([1.4570])

    Notes
    -----
    - theta_2(-z, q) = theta_2(z, q) (even function)
    - theta_2 is related to the Jacobi elliptic functions
    - The nome q is related to the elliptic modulus by q = exp(-pi*K'/K)
    """
    return torch.ops.torchscience.theta_2(z, q)
