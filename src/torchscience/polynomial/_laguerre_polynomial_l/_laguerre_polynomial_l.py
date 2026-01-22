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


class LaguerrePolynomialL(Tensor):
    """Laguerre series.

    Represents f(x) = sum_{k=0}^{n} c[k] * L_k(x)

    where L_k(x) are Laguerre polynomials.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of L_k(x).

    Notes
    -----
    The standard domain for Laguerre polynomials is [0, inf).
    The Laguerre polynomials are orthogonal with weight w(x) = exp(-x).
    """

    DOMAIN = (0.0, float("inf"))

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
        from ._laguerre_polynomial_l_evaluate import (
            laguerre_polynomial_l_evaluate,
        )

        return laguerre_polynomial_l_evaluate(self, x)

    def __add__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_add import laguerre_polynomial_l_add

        return laguerre_polynomial_l_add(self, other)

    def __radd__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_add import laguerre_polynomial_l_add

        return laguerre_polynomial_l_add(other, self)

    def __sub__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_subtract import (
            laguerre_polynomial_l_subtract,
        )

        return laguerre_polynomial_l_subtract(self, other)

    def __rsub__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_subtract import (
            laguerre_polynomial_l_subtract,
        )

        return laguerre_polynomial_l_subtract(other, self)

    def __neg__(self) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_negate import (
            laguerre_polynomial_l_negate,
        )

        return laguerre_polynomial_l_negate(self)

    def __mul__(
        self, other: Union["LaguerrePolynomialL", Tensor]
    ) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_multiply import (
            laguerre_polynomial_l_multiply,
        )
        from ._laguerre_polynomial_l_scale import (
            laguerre_polynomial_l_scale,
        )

        if isinstance(other, LaguerrePolynomialL):
            return laguerre_polynomial_l_multiply(self, other)
        return laguerre_polynomial_l_scale(self, other)

    def __rmul__(
        self, other: Union["LaguerrePolynomialL", Tensor]
    ) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_multiply import (
            laguerre_polynomial_l_multiply,
        )
        from ._laguerre_polynomial_l_scale import (
            laguerre_polynomial_l_scale,
        )

        if isinstance(other, LaguerrePolynomialL):
            return laguerre_polynomial_l_multiply(other, self)
        return laguerre_polynomial_l_scale(self, other)

    def __pow__(self, n: int) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_pow import laguerre_polynomial_l_pow

        return laguerre_polynomial_l_pow(self, n)

    def __floordiv__(
        self, other: "LaguerrePolynomialL"
    ) -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_div import laguerre_polynomial_l_div

        return laguerre_polynomial_l_div(self, other)

    def __mod__(self, other: "LaguerrePolynomialL") -> "LaguerrePolynomialL":
        from ._laguerre_polynomial_l_mod import laguerre_polynomial_l_mod

        return laguerre_polynomial_l_mod(self, other)

    def __repr__(self) -> str:
        return f"LaguerrePolynomialL({Tensor.__repr__(self)})"


def laguerre_polynomial_l(coeffs: Tensor) -> LaguerrePolynomialL:
    """Create Laguerre series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of L_k(x).
        Must have at least one coefficient.

    Returns
    -------
    LaguerrePolynomialL
        Laguerre series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))  # 1*L_0 + 2*L_1 + 3*L_2
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Laguerre series must have at least one coefficient"
        )

    return LaguerrePolynomialL(coeffs)
