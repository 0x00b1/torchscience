from typing import Union

import torch
from torch import Tensor

from torchscience.polynomial._polynomial_error import PolynomialError

# Operations that preserve polynomial type
_SHAPE_PRESERVING_OPS = {
    torch.clone,
    torch.detach,
    torch.Tensor.clone,
    torch.Tensor.detach,
    torch.Tensor.to,
    torch.Tensor.cuda,
    torch.Tensor.cpu,
    torch.Tensor.contiguous,
    torch.Tensor.requires_grad_,
}

_POLYNOMIAL_RETURNING_OPS = {
    torch.stack,
    torch.cat,
    torch.Tensor.__getitem__,
    torch.Tensor.reshape,
    torch.Tensor.view,
    torch.Tensor.squeeze,
    torch.Tensor.unsqueeze,
}


class Polynomial(Tensor):
    """Polynomial in power basis - IS the coefficients tensor.

    Represents p(x) = c[..., 0] + c[..., 1]*x + c[..., 2]*x^2 + ...

    Shape: (...batch, N) where N = degree + 1
    p[..., i] is the coefficient of x^i

    Examples
    --------
    Single polynomial 1 + 2x + 3x^2:
        polynomial(torch.tensor([1.0, 2.0, 3.0]))

    Batch of 2 polynomials:
        polynomial(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        # First: 1 + 2x, Second: 3 + 4x

    Operator overloading:
        p + q    # polynomial_add(p, q)
        p - q    # polynomial_subtract(p, q)
        p * q    # polynomial_multiply(p, q)
        -p       # polynomial_negate(p)
        p(x)     # polynomial_evaluate(p, x)
    """

    DOMAIN = (-float("inf"), float("inf"))

    @staticmethod
    def __new__(cls, data, *, dtype=None, device=None):
        if isinstance(data, Tensor):
            tensor = data.detach().clone()
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            if device is not None:
                tensor = tensor.to(device=device)
        else:
            tensor = torch.as_tensor(data, dtype=dtype, device=device)
        return tensor.as_subclass(cls)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = super().__torch_function__(func, types, args, kwargs)

        if func in _SHAPE_PRESERVING_OPS | _POLYNOMIAL_RETURNING_OPS:
            if isinstance(result, Tensor) and not isinstance(result, cls):
                result = result.as_subclass(cls)

        return result

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

    def __repr__(self) -> str:
        return f"Polynomial({super().__repr__()})"


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
    >>> p[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError("Polynomial must have at least one coefficient")

    return Polynomial(coeffs)
