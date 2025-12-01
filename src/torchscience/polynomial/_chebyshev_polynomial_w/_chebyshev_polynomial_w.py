from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class ChebyshevPolynomialW:
    """Chebyshev series of the fourth kind.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * W_k(x)

    where W_k(x) are Chebyshev polynomials of the fourth kind.

    The Chebyshev polynomials of the fourth kind are defined by:
        W_n(x) = sin((n + 1/2)θ) / sin(θ/2)  where x = cos(θ)

    They satisfy the recurrence relation:
        W_0(x) = 1
        W_1(x) = 2x + 1
        W_{n+1}(x) = 2x * W_n(x) - W_{n-1}(x)

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of W_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].

    The Chebyshev W polynomials are orthogonal on [-1, 1] with weight
    w(x) = sqrt((1-x)/(1+x)).
    """

    coeffs: Tensor

    DOMAIN = (-1.0, 1.0)

    def __call__(self, x: Tensor) -> Tensor:
        from ._chebyshev_polynomial_w_evaluate import (
            chebyshev_polynomial_w_evaluate,
        )

        return chebyshev_polynomial_w_evaluate(self, x)

    def __add__(self, other: "ChebyshevPolynomialW") -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_add import chebyshev_polynomial_w_add

        return chebyshev_polynomial_w_add(self, other)

    def __radd__(
        self, other: "ChebyshevPolynomialW"
    ) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_add import chebyshev_polynomial_w_add

        return chebyshev_polynomial_w_add(other, self)

    def __sub__(self, other: "ChebyshevPolynomialW") -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_subtract import (
            chebyshev_polynomial_w_subtract,
        )

        return chebyshev_polynomial_w_subtract(self, other)

    def __rsub__(
        self, other: "ChebyshevPolynomialW"
    ) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_subtract import (
            chebyshev_polynomial_w_subtract,
        )

        return chebyshev_polynomial_w_subtract(other, self)

    def __neg__(self) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_negate import (
            chebyshev_polynomial_w_negate,
        )

        return chebyshev_polynomial_w_negate(self)

    def __mul__(self, other):
        if isinstance(other, ChebyshevPolynomialW):
            from ._chebyshev_polynomial_w_multiply import (
                chebyshev_polynomial_w_multiply,
            )

            return chebyshev_polynomial_w_multiply(self, other)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_w_scale import (
                chebyshev_polynomial_w_scale,
            )

            return chebyshev_polynomial_w_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ChebyshevPolynomialW):
            from ._chebyshev_polynomial_w_multiply import (
                chebyshev_polynomial_w_multiply,
            )

            return chebyshev_polynomial_w_multiply(other, self)
        if isinstance(other, Tensor):
            from ._chebyshev_polynomial_w_scale import (
                chebyshev_polynomial_w_scale,
            )

            return chebyshev_polynomial_w_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_pow import chebyshev_polynomial_w_pow

        return chebyshev_polynomial_w_pow(self, n)

    def __floordiv__(
        self, other: "ChebyshevPolynomialW"
    ) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_div import chebyshev_polynomial_w_div

        return chebyshev_polynomial_w_div(self, other)

    def __mod__(self, other: "ChebyshevPolynomialW") -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_mod import chebyshev_polynomial_w_mod

        return chebyshev_polynomial_w_mod(self, other)


def chebyshev_polynomial_w(coeffs: Tensor) -> ChebyshevPolynomialW:
    """Create Chebyshev series of the fourth kind from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of W_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevPolynomialW
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_polynomial_w(torch.tensor([1.0, 2.0, 3.0]))  # 1*W_0 + 2*W_1 + 3*W_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialW(coeffs=coeffs)
