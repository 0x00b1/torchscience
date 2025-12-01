from torch import Tensor

from ._polynomial import Polynomial
from ._polynomial_antiderivative import polynomial_antiderivative
from ._polynomial_evaluate import polynomial_evaluate


def polynomial_integral(p: Polynomial, a: Tensor, b: Tensor) -> Tensor:
    """Compute definite integral.

    Parameters
    ----------
    p : Polynomial
        Polynomial to integrate.
    a, b : Tensor
        Integration bounds, broadcast with batch dimensions.

    Returns
    -------
    Tensor
        Definite integral integral_a^b p(x) dx.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # 1 + x^2
    >>> polynomial_integral(p, torch.tensor(0.0), torch.tensor(1.0))
    tensor(1.3333)  # 1 + 1/3
    """
    # Compute antiderivative (with constant 0)
    anti = polynomial_antiderivative(p, 0.0)

    # Evaluate at bounds and subtract
    return polynomial_evaluate(anti, b) - polynomial_evaluate(anti, a)
