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


class ChebyshevPolynomialV(Tensor):
    """Chebyshev series of the third kind.

    Represents f(x) = sum_{k=0}^{n} c[k] * V_k(x)

    where V_k(x) are Chebyshev polynomials of the third kind.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of V_k(x).

    The Chebyshev polynomials of the third kind are defined by:
        V_n(x) = cos((n + 1/2)theta) / cos(theta/2)  where x = cos(theta)

    They satisfy the recurrence relation:
        V_0(x) = 1
        V_1(x) = 2x - 1
        V_{n+1}(x) = 2x * V_n(x) - V_{n-1}(x)

    Notes
    -----
    The standard domain for Chebyshev polynomials is [-1, 1].

    The Chebyshev V polynomials are orthogonal on [-1, 1] with weight
    w(x) = sqrt((1+x)/(1-x)).
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
        from ._chebyshev_polynomial_v_evaluate import (
            chebyshev_polynomial_v_evaluate,
        )

        return chebyshev_polynomial_v_evaluate(self, x)

    def __add__(self, other: "ChebyshevPolynomialV") -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_add import chebyshev_polynomial_v_add

        return chebyshev_polynomial_v_add(self, other)

    def __radd__(
        self, other: "ChebyshevPolynomialV"
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_add import chebyshev_polynomial_v_add

        return chebyshev_polynomial_v_add(other, self)

    def __sub__(self, other: "ChebyshevPolynomialV") -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_subtract import (
            chebyshev_polynomial_v_subtract,
        )

        return chebyshev_polynomial_v_subtract(self, other)

    def __rsub__(
        self, other: "ChebyshevPolynomialV"
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_subtract import (
            chebyshev_polynomial_v_subtract,
        )

        return chebyshev_polynomial_v_subtract(other, self)

    def __neg__(self) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_negate import (
            chebyshev_polynomial_v_negate,
        )

        return chebyshev_polynomial_v_negate(self)

    def __mul__(
        self, other: Union["ChebyshevPolynomialV", Tensor]
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_multiply import (
            chebyshev_polynomial_v_multiply,
        )
        from ._chebyshev_polynomial_v_scale import (
            chebyshev_polynomial_v_scale,
        )

        if isinstance(other, ChebyshevPolynomialV):
            return chebyshev_polynomial_v_multiply(self, other)
        return chebyshev_polynomial_v_scale(self, other)

    def __rmul__(
        self, other: Union["ChebyshevPolynomialV", Tensor]
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_multiply import (
            chebyshev_polynomial_v_multiply,
        )
        from ._chebyshev_polynomial_v_scale import (
            chebyshev_polynomial_v_scale,
        )

        if isinstance(other, ChebyshevPolynomialV):
            return chebyshev_polynomial_v_multiply(other, self)
        return chebyshev_polynomial_v_scale(self, other)

    def __pow__(self, n: int) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_pow import chebyshev_polynomial_v_pow

        return chebyshev_polynomial_v_pow(self, n)

    def __floordiv__(
        self, other: "ChebyshevPolynomialV"
    ) -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_div import chebyshev_polynomial_v_div

        return chebyshev_polynomial_v_div(self, other)

    def __mod__(self, other: "ChebyshevPolynomialV") -> "ChebyshevPolynomialV":
        from ._chebyshev_polynomial_v_mod import chebyshev_polynomial_v_mod

        return chebyshev_polynomial_v_mod(self, other)

    def __repr__(self) -> str:
        return f"ChebyshevPolynomialV({Tensor.__repr__(self)})"


def chebyshev_polynomial_v(coeffs: Tensor) -> ChebyshevPolynomialV:
    """Create Chebyshev series of the third kind from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of V_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevPolynomialV
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([1.0, 2.0, 3.0]))  # 1*V_0 + 2*V_1 + 3*V_2
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialV(coeffs)
