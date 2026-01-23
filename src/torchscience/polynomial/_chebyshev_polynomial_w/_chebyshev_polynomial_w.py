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


class ChebyshevPolynomialW(Tensor):
    """Chebyshev series of the fourth kind.

    Represents f(x) = sum_{k=0}^{n} c[k] * W_k(x)

    where W_k(x) are Chebyshev polynomials of the fourth kind.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of W_k(x).

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].
    """

    DOMAIN = (-1.0, 1.0)

    @staticmethod
    def __new__(cls, data, *, dtype=None, device=None):
        if isinstance(data, Tensor):
            tensor = data.clone()
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

    def __mul__(
        self, other: Union["ChebyshevPolynomialW", Tensor]
    ) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_multiply import (
            chebyshev_polynomial_w_multiply,
        )
        from ._chebyshev_polynomial_w_scale import (
            chebyshev_polynomial_w_scale,
        )

        if isinstance(other, ChebyshevPolynomialW):
            return chebyshev_polynomial_w_multiply(self, other)
        return chebyshev_polynomial_w_scale(self, other)

    def __rmul__(
        self, other: Union["ChebyshevPolynomialW", Tensor]
    ) -> "ChebyshevPolynomialW":
        from ._chebyshev_polynomial_w_multiply import (
            chebyshev_polynomial_w_multiply,
        )
        from ._chebyshev_polynomial_w_scale import (
            chebyshev_polynomial_w_scale,
        )

        if isinstance(other, ChebyshevPolynomialW):
            return chebyshev_polynomial_w_multiply(other, self)
        return chebyshev_polynomial_w_scale(self, other)

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

    def __repr__(self) -> str:
        return f"ChebyshevPolynomialW({Tensor.__repr__(self)})"


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
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialW(coeffs)
