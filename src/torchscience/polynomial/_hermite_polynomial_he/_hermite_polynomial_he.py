"""HermitePolynomialHe tensorclass for Probabilists' Hermite series."""

from __future__ import annotations

from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError


@tensorclass
class HermitePolynomialHe:
    """Probabilists' Hermite polynomial series (He_n convention).

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * He_k(x)

    where He_k(x) are probabilists' Hermite polynomials.

    The probabilists' Hermite polynomials are orthogonal on (-inf, inf) with weight
    w(x) = exp(-x^2/2).

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of He_k(x).
        Batch dimensions come first, coefficient dimension last.

    Notes
    -----
    The standard domain for probabilists' Hermite polynomials is (-inf, inf).

    The three-term recurrence relation is:
        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

    The probabilists' Hermite polynomials are related to the physicists' version by:
        He_n(x) = 2^{-n/2} * H_n(x / sqrt(2))
    """

    coeffs: Tensor

    DOMAIN = (float("-inf"), float("inf"))

    def __call__(self, x: Tensor) -> Tensor:
        from ._hermite_polynomial_he_evaluate import (
            hermite_polynomial_he_evaluate,
        )

        return hermite_polynomial_he_evaluate(self, x)

    def __add__(self, other: "HermitePolynomialHe") -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_add import hermite_polynomial_he_add

        return hermite_polynomial_he_add(self, other)

    def __radd__(self, other: "HermitePolynomialHe") -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_add import hermite_polynomial_he_add

        return hermite_polynomial_he_add(other, self)

    def __sub__(self, other: "HermitePolynomialHe") -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_subtract import (
            hermite_polynomial_he_subtract,
        )

        return hermite_polynomial_he_subtract(self, other)

    def __rsub__(self, other: "HermitePolynomialHe") -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_subtract import (
            hermite_polynomial_he_subtract,
        )

        return hermite_polynomial_he_subtract(other, self)

    def __neg__(self) -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_negate import (
            hermite_polynomial_he_negate,
        )

        return hermite_polynomial_he_negate(self)

    def __mul__(self, other):
        if isinstance(other, HermitePolynomialHe):
            from ._hermite_polynomial_he_multiply import (
                hermite_polynomial_he_multiply,
            )

            return hermite_polynomial_he_multiply(self, other)
        if isinstance(other, Tensor):
            from ._hermite_polynomial_he_scale import (
                hermite_polynomial_he_scale,
            )

            return hermite_polynomial_he_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, HermitePolynomialHe):
            from ._hermite_polynomial_he_multiply import (
                hermite_polynomial_he_multiply,
            )

            return hermite_polynomial_he_multiply(other, self)
        if isinstance(other, Tensor):
            from ._hermite_polynomial_he_scale import (
                hermite_polynomial_he_scale,
            )

            return hermite_polynomial_he_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_pow import hermite_polynomial_he_pow

        return hermite_polynomial_he_pow(self, n)

    def __floordiv__(
        self, other: "HermitePolynomialHe"
    ) -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_div import hermite_polynomial_he_div

        return hermite_polynomial_he_div(self, other)

    def __mod__(self, other: "HermitePolynomialHe") -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_mod import hermite_polynomial_he_mod

        return hermite_polynomial_he_mod(self, other)


def hermite_polynomial_he(coeffs: Tensor) -> HermitePolynomialHe:
    """Create Probabilists' Hermite series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of He_k(x).
        Must have at least one coefficient.

    Returns
    -------
    HermitePolynomialHe
        Hermite series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = hermite_polynomial_he(torch.tensor([1.0, 2.0, 3.0]))  # 1*He_0 + 2*He_1 + 3*He_2
    >>> c.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Hermite series must have at least one coefficient"
        )

    return HermitePolynomialHe(coeffs=coeffs)
