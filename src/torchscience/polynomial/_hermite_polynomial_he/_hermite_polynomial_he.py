"""HermitePolynomialHe Tensor subclass for Probabilists' Hermite series."""

from __future__ import annotations

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


class HermitePolynomialHe(Tensor):
    """Probabilists' Hermite polynomial series (He_n convention).

    Represents f(x) = sum_{k=0}^{n} c[k] * He_k(x)

    where He_k(x) are probabilists' Hermite polynomials.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of He_k(x).

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

    def __mul__(
        self, other: Union["HermitePolynomialHe", Tensor]
    ) -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_multiply import (
            hermite_polynomial_he_multiply,
        )
        from ._hermite_polynomial_he_scale import (
            hermite_polynomial_he_scale,
        )

        if isinstance(other, HermitePolynomialHe):
            return hermite_polynomial_he_multiply(self, other)
        return hermite_polynomial_he_scale(self, other)

    def __rmul__(
        self, other: Union["HermitePolynomialHe", Tensor]
    ) -> "HermitePolynomialHe":
        from ._hermite_polynomial_he_multiply import (
            hermite_polynomial_he_multiply,
        )
        from ._hermite_polynomial_he_scale import (
            hermite_polynomial_he_scale,
        )

        if isinstance(other, HermitePolynomialHe):
            return hermite_polynomial_he_multiply(other, self)
        return hermite_polynomial_he_scale(self, other)

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

    def __repr__(self) -> str:
        return f"HermitePolynomialHe({Tensor.__repr__(self)})"


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
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Hermite series must have at least one coefficient"
        )

    return HermitePolynomialHe(coeffs)
