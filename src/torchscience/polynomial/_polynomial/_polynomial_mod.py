from ._polynomial import Polynomial
from ._polynomial_divmod import polynomial_divmod


def polynomial_mod(p: Polynomial, q: Polynomial) -> Polynomial:
    """Return remainder of polynomial division.

    Convenience wrapper around polynomial_divmod that returns only the remainder.

    Parameters
    ----------
    p : Polynomial
        Dividend polynomial.
    q : Polynomial
        Divisor polynomial.

    Returns
    -------
    Polynomial
        Remainder of p / q.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 0.0, 1.0]))  # x^2 + 1
    >>> q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
    >>> polynomial_mod(p, q)  # remainder is 2
    Polynomial(tensor([2.]))
    """
    _, remainder = polynomial_divmod(p, q)
    return remainder
