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


class LegendrePolynomialP(Tensor):
    """Legendre series.

    Represents f(x) = sum_{k=0}^{n} c[k] * P_k(x)

    where P_k(x) are Legendre polynomials.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of P_k(x).

    Notes
    -----
    The standard domain for Legendre polynomials is [-1, 1].
    """

    DOMAIN = (-1.0, 1.0)

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

    def __call__(self, x: Tensor) -> Tensor:
        from ._legendre_polynomial_p_evaluate import (
            legendre_polynomial_p_evaluate,
        )

        return legendre_polynomial_p_evaluate(self, x)

    def __add__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_add import legendre_polynomial_p_add

        return legendre_polynomial_p_add(self, other)

    def __radd__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_add import legendre_polynomial_p_add

        return legendre_polynomial_p_add(other, self)

    def __sub__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_subtract import (
            legendre_polynomial_p_subtract,
        )

        return legendre_polynomial_p_subtract(self, other)

    def __rsub__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_subtract import (
            legendre_polynomial_p_subtract,
        )

        return legendre_polynomial_p_subtract(other, self)

    def __neg__(self) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_negate import (
            legendre_polynomial_p_negate,
        )

        return legendre_polynomial_p_negate(self)

    def __mul__(
        self, other: Union["LegendrePolynomialP", Tensor]
    ) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_multiply import (
            legendre_polynomial_p_multiply,
        )
        from ._legendre_polynomial_p_scale import (
            legendre_polynomial_p_scale,
        )

        if isinstance(other, LegendrePolynomialP):
            return legendre_polynomial_p_multiply(self, other)
        return legendre_polynomial_p_scale(self, other)

    def __rmul__(
        self, other: Union["LegendrePolynomialP", Tensor]
    ) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_multiply import (
            legendre_polynomial_p_multiply,
        )
        from ._legendre_polynomial_p_scale import (
            legendre_polynomial_p_scale,
        )

        if isinstance(other, LegendrePolynomialP):
            return legendre_polynomial_p_multiply(other, self)
        return legendre_polynomial_p_scale(self, other)

    def __pow__(self, n: int) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_pow import legendre_polynomial_p_pow

        return legendre_polynomial_p_pow(self, n)

    def __floordiv__(
        self, other: "LegendrePolynomialP"
    ) -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_div import legendre_polynomial_p_div

        return legendre_polynomial_p_div(self, other)

    def __mod__(self, other: "LegendrePolynomialP") -> "LegendrePolynomialP":
        from ._legendre_polynomial_p_mod import legendre_polynomial_p_mod

        return legendre_polynomial_p_mod(self, other)

    def __repr__(self) -> str:
        return f"LegendrePolynomialP({Tensor.__repr__(self)})"


def legendre_polynomial_p(coeffs: Tensor) -> LegendrePolynomialP:
    """Create Legendre series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of P_k(x).
        Must have at least one coefficient.

    Returns
    -------
    LegendrePolynomialP
        Legendre series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))  # 1*P_0 + 2*P_1 + 3*P_2
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Legendre series must have at least one coefficient"
        )

    return LegendrePolynomialP(coeffs)
