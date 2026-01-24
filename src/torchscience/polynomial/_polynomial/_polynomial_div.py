from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._polynomial import Polynomial


def polynomial_div(p: "Polynomial", q: "Polynomial") -> "Polynomial":
    """Return quotient of polynomial division.

    Convenience wrapper around polynomial_divmod that returns only the quotient.

    Parameters
    ----------
    p : Polynomial
        Dividend polynomial.
    q : Polynomial
        Divisor polynomial.

    Returns
    -------
    Polynomial
        Quotient of p / q.

    Examples
    --------
    >>> p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))  # x^3 - 1
    >>> q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
    >>> polynomial_div(p, q)
    Polynomial(tensor([1., 1., 1.]))
    """
    from ._polynomial_divmod import polynomial_divmod

    quotient, _ = polynomial_divmod(p, q)
    return quotient
