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


class HermitePolynomialH(Tensor):
    """Physicists' Hermite polynomial series (H_n convention).

    Represents f(x) = sum_{k=0}^{n} c[k] * H_k(x)

    where H_k(x) are physicists' Hermite polynomials.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of H_k(x).

    The physicists' Hermite polynomials are orthogonal on (-inf, inf) with weight
    w(x) = exp(-x^2).

    Notes
    -----
    The standard domain for physicists' Hermite polynomials is (-inf, inf).

    The three-term recurrence relation is:
        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
    """

    DOMAIN = (float("-inf"), float("inf"))

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
        from ._hermite_polynomial_h_evaluate import (
            hermite_polynomial_h_evaluate,
        )

        return hermite_polynomial_h_evaluate(self, x)

    def __add__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_add import hermite_polynomial_h_add

        return hermite_polynomial_h_add(self, other)

    def __radd__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_add import hermite_polynomial_h_add

        return hermite_polynomial_h_add(other, self)

    def __sub__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_subtract import (
            hermite_polynomial_h_subtract,
        )

        return hermite_polynomial_h_subtract(self, other)

    def __rsub__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_subtract import (
            hermite_polynomial_h_subtract,
        )

        return hermite_polynomial_h_subtract(other, self)

    def __neg__(self) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_negate import (
            hermite_polynomial_h_negate,
        )

        return hermite_polynomial_h_negate(self)

    def __mul__(
        self, other: Union["HermitePolynomialH", Tensor]
    ) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_multiply import (
            hermite_polynomial_h_multiply,
        )
        from ._hermite_polynomial_h_scale import (
            hermite_polynomial_h_scale,
        )

        if isinstance(other, HermitePolynomialH):
            return hermite_polynomial_h_multiply(self, other)
        return hermite_polynomial_h_scale(self, other)

    def __rmul__(
        self, other: Union["HermitePolynomialH", Tensor]
    ) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_multiply import (
            hermite_polynomial_h_multiply,
        )
        from ._hermite_polynomial_h_scale import (
            hermite_polynomial_h_scale,
        )

        if isinstance(other, HermitePolynomialH):
            return hermite_polynomial_h_multiply(other, self)
        return hermite_polynomial_h_scale(self, other)

    def __pow__(self, n: int) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_pow import hermite_polynomial_h_pow

        return hermite_polynomial_h_pow(self, n)

    def __floordiv__(
        self, other: "HermitePolynomialH"
    ) -> "HermitePolynomialH":
        from ._hermite_polynomial_h_div import hermite_polynomial_h_div

        return hermite_polynomial_h_div(self, other)

    def __mod__(self, other: "HermitePolynomialH") -> "HermitePolynomialH":
        from ._hermite_polynomial_h_mod import hermite_polynomial_h_mod

        return hermite_polynomial_h_mod(self, other)

    def __repr__(self) -> str:
        return f"HermitePolynomialH({Tensor.__repr__(self)})"


def hermite_polynomial_h(coeffs: Tensor) -> HermitePolynomialH:
    """Create Physicists' Hermite series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of H_k(x).
        Must have at least one coefficient.

    Returns
    -------
    HermitePolynomialH
        Hermite series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([1.0, 2.0, 3.0]))  # 1*H_0 + 2*H_1 + 3*H_2
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Hermite series must have at least one coefficient"
        )

    return HermitePolynomialH(coeffs)
