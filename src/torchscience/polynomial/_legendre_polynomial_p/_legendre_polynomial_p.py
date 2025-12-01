from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class LegendrePolynomialP:
    """Legendre series.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * P_k(x)

    where P_k(x) are Legendre polynomials.

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of P_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Legendre polynomials is [-1, 1].
    """

    coeffs: Tensor

    DOMAIN = (-1.0, 1.0)

    def __call__(self, x: Tensor) -> Tensor:
        from ._legendre_polynomial_p_evaluate import (
            legendre_polynomial_p_evaluate,
        )

        return legendre_polynomial_p_evaluate(self, x)

    def __add__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_add import legendre_polynomial_p_add

        return legendre_polynomial_p_add(self, other)

    def __radd__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_add import legendre_polynomial_p_add

        return legendre_polynomial_p_add(other, self)

    def __sub__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_subtract import (
            legendre_polynomial_p_subtract,
        )

        return legendre_polynomial_p_subtract(self, other)

    def __rsub__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_subtract import (
            legendre_polynomial_p_subtract,
        )

        return legendre_polynomial_p_subtract(other, self)

    def __neg__(self) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_negate import (
            legendre_polynomial_p_negate,
        )

        return legendre_polynomial_p_negate(self)

    def __mul__(self, other):
        if isinstance(other, LegendrePolynomialP):
            from ._legendre_polynomial_p_multiply import (
                legendre_polynomial_p_multiply,
            )

            return legendre_polynomial_p_multiply(self, other)
        if isinstance(other, Tensor):
            from ._legendre_polynomial_p_scale import (
                legendre_polynomial_p_scale,
            )

            return legendre_polynomial_p_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, LegendrePolynomialP):
            from ._legendre_polynomial_p_multiply import (
                legendre_polynomial_p_multiply,
            )

            return legendre_polynomial_p_multiply(other, self)
        if isinstance(other, Tensor):
            from ._legendre_polynomial_p_scale import (
                legendre_polynomial_p_scale,
            )

            return legendre_polynomial_p_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_pow import legendre_polynomial_p_pow

        return legendre_polynomial_p_pow(self, n)

    def __floordiv__(
        self, other: "LegendrePolynomialP"
    ) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_div import legendre_polynomial_p_div

        return legendre_polynomial_p_div(self, other)

    def __mod__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_mod import legendre_polynomial_p_mod

        return legendre_polynomial_p_mod(self, other)


def legendre_polynomial_p(coeffs: Tensor) -> LegendrePolynomialP:
    """Create Legendre series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of P_k(x).
        Must have at least one coefficient.

    Returns
    -------
    LegendrePolynomialP
        Legendre series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))  # 1*P_0 + 2*P_1 + 3*P_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Legendre series must have at least one coefficient"
        )

    return LegendrePolynomialP(coeffs=coeffs)
