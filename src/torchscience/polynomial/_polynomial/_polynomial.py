from typing import TYPE_CHECKING, Union

from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError

if TYPE_CHECKING:
    pass


@tensorclass
class Polynomial:
    """Polynomial in power basis with ascending coefficients.

    Represents p(x) = coeffs[..., 0] + coeffs[..., 1]*x + coeffs[..., 2]*x^2 + ...

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., i] is the coefficient of x^i.
        Batch dimensions come first, coefficient dimension last.

    Examples
    --------
    Single polynomial 1 + 2x + 3x^2:
        Polynomial(coeffs=torch.tensor([1.0, 2.0, 3.0]))

    Batch of 2 polynomials:
        Polynomial(coeffs=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        # First: 1 + 2x, Second: 3 + 4x

    Operator overloading:
        p + q    # polynomial_add(p, q)
        p - q    # polynomial_subtract(p, q)
        p * q    # polynomial_multiply(p, q)
        -p       # polynomial_negate(p)
        p(x)     # polynomial_evaluate(p, x)
    """

    coeffs: Tensor

    def __add__(self, other: "Polynomial") -> "Polynomial":
        from ._polynomial_add import polynomial_add

        return polynomial_add(self, other)

    def __radd__(self, other: "Polynomial") -> "Polynomial":
        from ._polynomial_add import polynomial_add

        return polynomial_add(other, self)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        from ._polynomial_subtract import polynomial_subtract

        return polynomial_subtract(self, other)

    def __rsub__(self, other: "Polynomial") -> "Polynomial":
        from ._polynomial_subtract import polynomial_subtract

        return polynomial_subtract(other, self)

    def __mul__(self, other: Union["Polynomial", Tensor]) -> "Polynomial":
        from ._polynomial_multiply import polynomial_multiply
        from ._polynomial_scale import polynomial_scale

        if isinstance(other, Polynomial):
            return polynomial_multiply(self, other)
        return polynomial_scale(self, other)

    def __rmul__(self, other: Union["Polynomial", Tensor]) -> "Polynomial":
        from ._polynomial_multiply import polynomial_multiply
        from ._polynomial_scale import polynomial_scale

        if isinstance(other, Polynomial):
            return polynomial_multiply(other, self)
        return polynomial_scale(self, other)

    def __neg__(self) -> "Polynomial":
        from ._polynomial_negate import polynomial_negate

        return polynomial_negate(self)

    def __call__(self, x: Tensor) -> Tensor:
        from ._polynomial_evaluate import polynomial_evaluate

        return polynomial_evaluate(self, x)

    def __pow__(self, n: int) -> "Polynomial":
        from ._polynomial_pow import polynomial_pow

        return polynomial_pow(self, n)

    def __floordiv__(self, other: "Polynomial") -> "Polynomial":
        from ._polynomial_div import polynomial_div

        return polynomial_div(self, other)

    def __mod__(self, other: "Polynomial") -> "Polynomial":
        from ._polynomial_mod import polynomial_mod

        return polynomial_mod(self, other)


def polynomial(coeffs: Tensor) -> Polynomial:
    """Create polynomial from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        Must have at least one coefficient.

    Returns
    -------
    Polynomial
        Polynomial instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> p.coeffs
    tensor([1., 2., 3.])
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError("Polynomial must have at least one coefficient")

    return Polynomial(coeffs=coeffs)
