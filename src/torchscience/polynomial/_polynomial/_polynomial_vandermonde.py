import torch
from torch import Tensor


def polynomial_vandermonde(x: Tensor, degree: int) -> Tensor:
    """Construct Vandermonde matrix for polynomial fitting.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n_points,).
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    Tensor
        Vandermonde matrix, shape (n_points, degree + 1).
        V[i, j] = x[i]^j (ascending powers).

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> polynomial_vandermonde(x, 2)
    tensor([[1., 1., 1.],
            [1., 2., 4.],
            [1., 3., 9.]])
    """
    n = x.shape[0]
    powers = torch.arange(degree + 1, dtype=x.dtype, device=x.device)
    # x[:, None] ** powers[None, :] gives (n, degree+1)
    return x.unsqueeze(-1) ** powers.unsqueeze(0)
