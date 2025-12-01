from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class ChebyshevPolynomialV:
    """Chebyshev series of the third kind.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * V_k(x)

    where V_k(x) are Chebyshev polynomials of the third kind.

    The Chebyshev polynomials of the third kind are defined by:
        V_n(x) = cos((n + 1/2)θ) / cos(θ/2)  where x = cos(θ)

    They satisfy the recurrence relation:
        V_0(x) = 1
        V_1(x) = 2x - 1
        V_{n+1}(x) = 2x * V_n(x) - V_{n-1}(x)

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of V_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].

    The Chebyshev V polynomials are orthogonal on [-1, 1] with weight
    w(x) = sqrt((1+x)/(1-x)).
    """

    coeffs: Tensor

    DOMAIN = (-1.0, 1.0)

    def __call__(self, x: Tensor) -> Tensor:
        from ._chebyshev_polynomial_v_evaluate import (
            chebyshev_polynomial_v_evaluate,
        )

        return chebyshev_polynomial_v_evaluate(self, x)

    def __add__(self, other: "ChebyshevPolynomialV") -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_add import chebyshev_polynomial_v_add

        return chebyshev_polynomial_v_add(self, other)

    def __radd__(
        self, other: "ChebyshevPolynomialV"
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_add import chebyshev_polynomial_v_add

        return chebyshev_polynomial_v_add(other, self)

    def __sub__(self, other: "ChebyshevPolynomialV") -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_subtract import (
            chebyshev_polynomial_v_subtract,
        )

        return chebyshev_polynomial_v_subtract(self, other)

    def __rsub__(
        self, other: "ChebyshevPolynomialV"
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_subtract import (
            chebyshev_polynomial_v_subtract,
        )

        return chebyshev_polynomial_v_subtract(other, self)

    def __neg__(self) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_negate import (
            chebyshev_polynomial_v_negate,
        )

        return chebyshev_polynomial_v_negate(self)

    def __mul__(self, other):
        if isinstance(other, ChebyshevPolynomialV):
            from ._chebyshev_polynomial_v_multiply import (
                chebyshev_polynomial_v_multiply,
            )

            return chebyshev_polynomial_v_multiply(self, other)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_v_scale import (
                chebyshev_polynomial_v_scale,
            )

            return chebyshev_polynomial_v_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ChebyshevPolynomialV):
            from ._chebyshev_polynomial_v_multiply import (
                chebyshev_polynomial_v_multiply,
            )

            return chebyshev_polynomial_v_multiply(other, self)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_v_scale import (
                chebyshev_polynomial_v_scale,
            )

            return chebyshev_polynomial_v_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_pow import chebyshev_polynomial_v_pow

        return chebyshev_polynomial_v_pow(self, n)

    def __floordiv__(
        self, other: "ChebyshevPolynomialV"
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_div import chebyshev_polynomial_v_div

        return chebyshev_polynomial_v_div(self, other)

    def __mod__(self, other: "ChebyshevPolynomialV") -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_mod import chebyshev_polynomial_v_mod

        return chebyshev_polynomial_v_mod(self, other)


def chebyshev_polynomial_v(coeffs: Tensor) -> ChebyshevPolynomialV:
    """Create Chebyshev series of the third kind from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of V_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevPolynomialV
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([1.0, 2.0, 3.0]))  # 1*V_0 + 2*V_1 + 3*V_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialV(coeffs=coeffs)
