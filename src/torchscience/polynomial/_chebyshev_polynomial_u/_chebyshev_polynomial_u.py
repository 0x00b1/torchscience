from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class ChebyshevPolynomialU:
    """Chebyshev series of the second kind.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * U_k(x)

    where U_k(x) are Chebyshev polynomials of the second kind.

    The Chebyshev polynomials of the second kind satisfy:
        U_0(x) = 1
        U_1(x) = 2x
        U_{n+1}(x) = 2x * U_n(x) - U_{n-1}(x)

    They are orthogonal on [-1, 1] with weight w(x) = sqrt(1 - x^2).

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of U_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].
    """

    coeffs: Tensor

    DOMAIN = (-1.0, 1.0)

    def __call__(self, x: Tensor) -> Tensor:
        from ._chebyshev_polynomial_u_evaluate import (
            chebyshev_polynomial_u_evaluate,
        )

        return chebyshev_polynomial_u_evaluate(self, x)

    def __add__(self, other: "ChebyshevPolynomialU") -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_add import chebyshev_polynomial_u_add

        return chebyshev_polynomial_u_add(self, other)

    def __radd__(
        self, other: "ChebyshevPolynomialU"
    ) -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_add import chebyshev_polynomial_u_add

        return chebyshev_polynomial_u_add(other, self)

    def __sub__(self, other: "ChebyshevPolynomialU") -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_subtract import (
            chebyshev_polynomial_u_subtract,
        )

        return chebyshev_polynomial_u_subtract(self, other)

    def __rsub__(
        self, other: "ChebyshevPolynomialU"
    ) -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_subtract import (
            chebyshev_polynomial_u_subtract,
        )

        return chebyshev_polynomial_u_subtract(other, self)

    def __neg__(self) -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_negate import (
            chebyshev_polynomial_u_negate,
        )

        return chebyshev_polynomial_u_negate(self)

    def __mul__(self, other):
        if isinstance(other, ChebyshevPolynomialU):
            from ._chebyshev_polynomial_u_multiply import (
                chebyshev_polynomial_u_multiply,
            )

            return chebyshev_polynomial_u_multiply(self, other)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_u_scale import (
                chebyshev_polynomial_u_scale,
            )

            return chebyshev_polynomial_u_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ChebyshevPolynomialU):
            from ._chebyshev_polynomial_u_multiply import (
                chebyshev_polynomial_u_multiply,
            )

            return chebyshev_polynomial_u_multiply(other, self)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_u_scale import (
                chebyshev_polynomial_u_scale,
            )

            return chebyshev_polynomial_u_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_pow import chebyshev_polynomial_u_pow

        return chebyshev_polynomial_u_pow(self, n)

    def __floordiv__(
        self, other: "ChebyshevPolynomialU"
    ) -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_div import chebyshev_polynomial_u_div

        return chebyshev_polynomial_u_div(self, other)

    def __mod__(self, other: "ChebyshevPolynomialU") -> "ChebyshevPolynomialU":
        from ._chebyshev_polynomial_u_mod import chebyshev_polynomial_u_mod

        return chebyshev_polynomial_u_mod(self, other)


def chebyshev_polynomial_u(coeffs: Tensor) -> ChebyshevPolynomialU:
    """Create Chebyshev series of the second kind from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of U_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevPolynomialU
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_polynomial_u(torch.tensor([1.0, 2.0, 3.0]))  # 1*U_0 + 2*U_1 + 3*U_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialU(coeffs=coeffs)
