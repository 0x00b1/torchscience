import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_negate(
    a: HermitePolynomialH,
) -> HermitePolynomialH:
    """Negate a Physicists' Hermite series.

    Parameters
    ----------
    a : HermitePolynomialH
        Series to negate.

    Returns
    -------
    HermitePolynomialH
        Negated series -a.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = hermite_polynomial_h_negate(a)
    >>> b
    HermitePolynomialH(tensor([-1.,  2., -3.]))
    """
    result = torch.Tensor.neg(a)
    return hermite_polynomial_h(result)
