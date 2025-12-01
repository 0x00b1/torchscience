from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class ChebyshevPolynomialT:
    """Chebyshev series of the first kind.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * T_k(x)

    where T_k(x) are Chebyshev polynomials of the first kind.

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of T_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].
    """

    coeffs: Tensor

    DOMAIN = (-1.0, 1.0)

    def __call__(self, x: Tensor) -> Tensor:
        from ._chebyshev_polynomial_t_evaluate import (
            chebyshev_polynomial_t_evaluate,
        )

        return chebyshev_polynomial_t_evaluate(self, x)

    def __add__(self, other: "ChebyshevPolynomialT") -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_add import chebyshev_polynomial_t_add

        return chebyshev_polynomial_t_add(self, other)

    def __radd__(
        self, other: "ChebyshevPolynomialT"
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_add import chebyshev_polynomial_t_add

        return chebyshev_polynomial_t_add(other, self)

    def __sub__(self, other: "ChebyshevPolynomialT") -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_subtract import (
            chebyshev_polynomial_t_subtract,
        )

        return chebyshev_polynomial_t_subtract(self, other)

    def __rsub__(
        self, other: "ChebyshevPolynomialT"
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_subtract import (
            chebyshev_polynomial_t_subtract,
        )

        return chebyshev_polynomial_t_subtract(other, self)

    def __neg__(self) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_negate import (
            chebyshev_polynomial_t_negate,
        )

        return chebyshev_polynomial_t_negate(self)

    def __mul__(self, other):
        if isinstance(other, ChebyshevPolynomialT):
            from ._chebyshev_polynomial_t_multiply import (
                chebyshev_polynomial_t_multiply,
            )

            return chebyshev_polynomial_t_multiply(self, other)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_t_scale import (
                chebyshev_polynomial_t_scale,
            )

            return chebyshev_polynomial_t_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ChebyshevPolynomialT):
            from ._chebyshev_polynomial_t_multiply import (
                chebyshev_polynomial_t_multiply,
            )

            return chebyshev_polynomial_t_multiply(other, self)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_t_scale import (
                chebyshev_polynomial_t_scale,
            )

            return chebyshev_polynomial_t_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_pow import chebyshev_polynomial_t_pow

        return chebyshev_polynomial_t_pow(self, n)

    def __floordiv__(
        self, other: "ChebyshevPolynomialT"
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_div import chebyshev_polynomial_t_div

        return chebyshev_polynomial_t_div(self, other)

    def __mod__(self, other: "ChebyshevPolynomialT") -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_mod import chebyshev_polynomial_t_mod

        return chebyshev_polynomial_t_mod(self, other)


def chebyshev_polynomial_t(coeffs: Tensor) -> ChebyshevPolynomialT:
    """Create Chebyshev series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of T_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevPolynomialT
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))  # 1*T_0 + 2*T_1 + 3*T_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialT(coeffs=coeffs)
