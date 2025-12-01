from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class HermitePolynomialH:
    """Physicists' Hermite polynomial series (H_n convention).

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * H_k(x)

    where H_k(x) are physicists' Hermite polynomials.

    The physicists' Hermite polynomials are orthogonal on (-inf, inf) with weight
    w(x) = exp(-x^2).

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of H_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for physicists' Hermite polynomials is (-inf, inf).

    The three-term recurrence relation is:
        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
    """

    coeffs: Tensor

    DOMAIN = (float("-inf"), float("inf"))

    def __call__(self, x: Tensor) -> Tensor:
        from ._hermite_polynomial_h_evaluate import (
            hermite_polynomial_h_evaluate,
        )

        return hermite_polynomial_h_evaluate(self, x)

    def __add__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_add import hermite_polynomial_h_add

        return hermite_polynomial_h_add(self, other)

    def __radd__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_add import hermite_polynomial_h_add

        return hermite_polynomial_h_add(other, self)

    def __sub__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_subtract import (
            hermite_polynomial_h_subtract,
        )

        return hermite_polynomial_h_subtract(self, other)

    def __rsub__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_subtract import (
            hermite_polynomial_h_subtract,
        )

        return hermite_polynomial_h_subtract(other, self)

    def __neg__(self) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_negate import (
            hermite_polynomial_h_negate,
        )

        return hermite_polynomial_h_negate(self)

    def __mul__(self, other):
        if isinstance(other, HermitePolynomialH):
            from ._hermite_polynomial_h_multiply import (
                hermite_polynomial_h_multiply,
            )

            return hermite_polynomial_h_multiply(self, other)
        if isinstance(other, Tensor):
            from ._hermite_polynomial_h_scale import (
                hermite_polynomial_h_scale,
            )

            return hermite_polynomial_h_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, HermitePolynomialH):
            from ._hermite_polynomial_h_multiply import (
                hermite_polynomial_h_multiply,
            )

            return hermite_polynomial_h_multiply(other, self)
        if isinstance(other, Tensor):
            from ._hermite_polynomial_h_scale import (
                hermite_polynomial_h_scale,
            )

            return hermite_polynomial_h_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_pow import hermite_polynomial_h_pow

        return hermite_polynomial_h_pow(self, n)

    def __floordiv__(
        self, other: "HermitePolynomialH"
    ) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_div import hermite_polynomial_h_div

        return hermite_polynomial_h_div(self, other)

    def __mod__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_mod import hermite_polynomial_h_mod

        return hermite_polynomial_h_mod(self, other)


def hermite_polynomial_h(coeffs: Tensor) -> HermitePolynomialH:
    """Create Physicists' Hermite series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of H_k(x).
        Must have at least one coefficient.

    Returns
    -------
    HermitePolynomialH
        Hermite series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([1.0, 2.0, 3.0]))  # 1*H_0 + 2*H_1 + 3*H_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Hermite series must have at least one coefficient"
        )

    return HermitePolynomialH(coeffs=coeffs)
