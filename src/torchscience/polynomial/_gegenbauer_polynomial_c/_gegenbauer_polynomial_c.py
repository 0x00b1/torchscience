import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._exceptions import ParameterError
from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class GegenbauerPolynomialC:
    """Gegenbauer (ultraspherical) polynomial series.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * C_k^{lambda}(x)

    where C_k^{lambda}(x) are Gegenbauer polynomials with parameter lambda.

    The Gegenbauer polynomials are orthogonal on [-1, 1] with weight
    w(x) = (1 - x^2)^{lambda - 1/2}.

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of C_k^{lambda}(x).
        Batch dimensions come first, coefficient dimension last.
    lambda_ : Tensor
        Parameter lambda, must be > -1/2. Tensor for batch support.

    Notes
    -----
    The standard domain for Gegenbauer polynomials is [-1, 1].

    Special cases:
    - lambda = 1/2: Legendre polynomials (up to normalization)
    - lambda = 1: Chebyshev U polynomials (up to normalization)
    - lambda -> 0: Chebyshev T polynomials (as limit)
    """

    coeffs: Tensor
    lambda_: Tensor

    DOMAIN = (-1.0, 1.0)

    def __post_init__(self):
        if (self.lambda_ <= -0.5).any():
            raise ParameterError(f"lambda must be > -1/2, got {self.lambda_}")

    def __call__(self, x: Tensor) -> Tensor:
        from ._gegenbauer_polynomial_c_evaluate import (
            gegenbauer_polynomial_c_evaluate,
        )

        return gegenbauer_polynomial_c_evaluate(self, x)

    def __add__(
        self, other: "GegenbauerPolynomialC"
    ) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_add import gegenbauer_polynomial_c_add

        return gegenbauer_polynomial_c_add(self, other)

    def __radd__(
        self, other: "GegenbauerPolynomialC"
    ) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_add import gegenbauer_polynomial_c_add

        return gegenbauer_polynomial_c_add(other, self)

    def __sub__(
        self, other: "GegenbauerPolynomialC"
    ) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_subtract import (
            gegenbauer_polynomial_c_subtract,
        )

        return gegenbauer_polynomial_c_subtract(self, other)

    def __rsub__(
        self, other: "GegenbauerPolynomialC"
    ) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_subtract import (
            gegenbauer_polynomial_c_subtract,
        )

        return gegenbauer_polynomial_c_subtract(other, self)

    def __neg__(self) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_negate import (
            gegenbauer_polynomial_c_negate,
        )

        return gegenbauer_polynomial_c_negate(self)

    def __mul__(self, other):
        if isinstance(other, GegenbauerPolynomialC):
            from ._gegenbauer_polynomial_c_multiply import (
                gegenbauer_polynomial_c_multiply,
            )

            return gegenbauer_polynomial_c_multiply(self, other)
        if isinstance(other, Tensor):
            from ._gegenbauer_polynomial_c_scale import (
                gegenbauer_polynomial_c_scale,
            )

            return gegenbauer_polynomial_c_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, GegenbauerPolynomialC):
            from ._gegenbauer_polynomial_c_multiply import (
                gegenbauer_polynomial_c_multiply,
            )

            return gegenbauer_polynomial_c_multiply(other, self)
        if isinstance(other, Tensor):
            from ._gegenbauer_polynomial_c_scale import (
                gegenbauer_polynomial_c_scale,
            )

            return gegenbauer_polynomial_c_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_pow import gegenbauer_polynomial_c_pow

        return gegenbauer_polynomial_c_pow(self, n)

    def __floordiv__(
        self, other: "GegenbauerPolynomialC"
    ) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_div import gegenbauer_polynomial_c_div

        return gegenbauer_polynomial_c_div(self, other)

    def __mod__(
        self, other: "GegenbauerPolynomialC"
    ) -> "GegenbauerPolynomialC":
        from ._gegenbauer_polynomial_c_mod import gegenbauer_polynomial_c_mod

        return gegenbauer_polynomial_c_mod(self, other)


def gegenbauer_polynomial_c(
    coeffs: Tensor,
    lambda_: Tensor,
) -> GegenbauerPolynomialC:
    """Create Gegenbauer series from coefficient tensor and parameter.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of C_k^{lambda}(x).
        Must have at least one coefficient.
    lambda_ : Tensor
        Parameter lambda, must be > -1/2.

    Returns
    -------
    GegenbauerPolynomialC
        Gegenbauer series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).
    ParameterError
        If lambda <= -1/2.

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(
    ...     torch.tensor([1.0, 2.0, 3.0]),
    ...     torch.tensor(1.0)
    ... )  # 1*C_0^1 + 2*C_1^1 + 3*C_2^1
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Gegenbauer series must have at least one coefficient"
        )

    # Convert lambda_ to tensor if scalar
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(
            lambda_, dtype=coeffs.dtype, device=coeffs.device
        )

    return GegenbauerPolynomialC(coeffs=coeffs, lambda_=lambda_)
