from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class LaguerrePolynomialL:
    """Laguerre series.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * L_k(x)

    where L_k(x) are Laguerre polynomials.

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of L_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Laguerre polynomials is [0, ∞).
    The Laguerre polynomials are orthogonal with weight w(x) = exp(-x).
    """

    coeffs: Tensor

    DOMAIN = (0.0, float("inf"))

    def __call__(self, x: Tensor) -> Tensor:
        from ._laguerre_polynomial_l_evaluate import (
            laguerre_polynomial_l_evaluate,
        )

        return laguerre_polynomial_l_evaluate(self, x)

    def __add__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_add import laguerre_polynomial_l_add

        return laguerre_polynomial_l_add(self, other)

    def __radd__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_add import laguerre_polynomial_l_add

        return laguerre_polynomial_l_add(other, self)

    def __sub__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_subtract import (
            laguerre_polynomial_l_subtract,
        )

        return laguerre_polynomial_l_subtract(self, other)

    def __rsub__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_subtract import (
            laguerre_polynomial_l_subtract,
        )

        return laguerre_polynomial_l_subtract(other, self)

    def __neg__(self) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_negate import (
            laguerre_polynomial_l_negate,
        )

        return laguerre_polynomial_l_negate(self)

    def __mul__(self, other):
        if isinstance(other, LaguerrePolynomialL):
            from ._laguerre_polynomial_l_multiply import (
                laguerre_polynomial_l_multiply,
            )

            return laguerre_polynomial_l_multiply(self, other)
        if isinstance(other, Tensor):
            from ._laguerre_polynomial_l_scale import (
                laguerre_polynomial_l_scale,
            )

            return laguerre_polynomial_l_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, LaguerrePolynomialL):
            from ._laguerre_polynomial_l_multiply import (
                laguerre_polynomial_l_multiply,
            )

            return laguerre_polynomial_l_multiply(other, self)
        if isinstance(other, Tensor):
            from ._laguerre_polynomial_l_scale import (
                laguerre_polynomial_l_scale,
            )

            return laguerre_polynomial_l_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_pow import laguerre_polynomial_l_pow

        return laguerre_polynomial_l_pow(self, n)

    def __floordiv__(
        self, other: "LaguerrePolynomialL"
    ) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_div import laguerre_polynomial_l_div

        return laguerre_polynomial_l_div(self, other)

    def __mod__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_mod import laguerre_polynomial_l_mod

        return laguerre_polynomial_l_mod(self, other)


def laguerre_polynomial_l(coeffs: Tensor) -> LaguerrePolynomialL:
    """Create Laguerre series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of L_k(x).
        Must have at least one coefficient.

    Returns
    -------
    LaguerrePolynomialL
        Laguerre series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))  # 1*L_0 + 2*L_1 + 3*L_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Laguerre series must have at least one coefficient"
        )

    return LaguerrePolynomialL(coeffs=coeffs)
