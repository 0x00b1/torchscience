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


class ChebyshevPolynomialT(Tensor):
    """Chebyshev series of the first kind.

    Represents f(x) = sum_{k=0}^{n} c[k] * T_k(x)

    where T_k(x) are Chebyshev polynomials of the first kind.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of T_k(x).

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
        from ._chebyshev_polynomial_t_evaluate import (
            chebyshev_polynomial_t_evaluate,
        )

        return chebyshev_polynomial_t_evaluate(self, x)

    def __add__(
        self, other: Union["ChebyshevPolynomialT", Tensor]
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_add import chebyshev_polynomial_t_add

        # Check if both are proper polynomials (at least 1D with coefficients)
        if (
            self.dim() >= 1
            and isinstance(other, ChebyshevPolynomialT)
            and other.dim() >= 1
        ):
            return chebyshev_polynomial_t_add(self, other)
        # Handle scalar cases: if self is 0-dim, just do tensor add
        if self.dim() == 0:
            return self.as_subclass(Tensor) + other
        # Scalar addition to polynomial: add to constant term only
        result = self.clone()
        result[..., 0] = result[..., 0].as_subclass(Tensor) + other
        return result

    def __radd__(
        self, other: Union["ChebyshevPolynomialT", Tensor]
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_add import chebyshev_polynomial_t_add

        # Check if both are proper polynomials (at least 1D with coefficients)
        if (
            self.dim() >= 1
            and isinstance(other, ChebyshevPolynomialT)
            and other.dim() >= 1
        ):
            return chebyshev_polynomial_t_add(other, self)
        # Handle scalar cases: if self is 0-dim, just do tensor add
        if self.dim() == 0:
            return self.as_subclass(Tensor) + other
        # Scalar addition to polynomial: add to constant term only
        result = self.clone()
        result[..., 0] = result[..., 0].as_subclass(Tensor) + other
        return result

    def __sub__(
        self, other: Union["ChebyshevPolynomialT", Tensor]
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_subtract import (
            chebyshev_polynomial_t_subtract,
        )

        # Check if both are proper polynomials (at least 1D with coefficients)
        if (
            self.dim() >= 1
            and isinstance(other, ChebyshevPolynomialT)
            and other.dim() >= 1
        ):
            return chebyshev_polynomial_t_subtract(self, other)
        # Handle scalar cases: if self is 0-dim, just do tensor sub
        if self.dim() == 0:
            return self.as_subclass(Tensor) - other
        # Scalar subtraction from polynomial: subtract from constant term only
        result = self.clone()
        result[..., 0] = result[..., 0].as_subclass(Tensor) - other
        return result

    def __rsub__(
        self, other: Union["ChebyshevPolynomialT", Tensor]
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_subtract import (
            chebyshev_polynomial_t_subtract,
        )

        # Check if both are proper polynomials (at least 1D with coefficients)
        if (
            self.dim() >= 1
            and isinstance(other, ChebyshevPolynomialT)
            and other.dim() >= 1
        ):
            return chebyshev_polynomial_t_subtract(other, self)
        # Handle scalar cases: if self is 0-dim, just do tensor sub
        if self.dim() == 0:
            # Convert Python scalar to tensor if needed
            if not isinstance(other, Tensor):
                other = torch.as_tensor(
                    other, dtype=self.dtype, device=self.device
                )
            return other - self.as_subclass(Tensor)
        # Scalar - polynomial: negate self and add scalar to constant term
        result = -self
        result[..., 0] = result[..., 0].as_subclass(Tensor) + other
        return result

    def __neg__(self) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_negate import (
            chebyshev_polynomial_t_negate,
        )

        return chebyshev_polynomial_t_negate(self)

    def __mul__(
        self, other: Union["ChebyshevPolynomialT", Tensor]
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_multiply import (
            chebyshev_polynomial_t_multiply,
        )
        from ._chebyshev_polynomial_t_scale import (
            chebyshev_polynomial_t_scale,
        )

        # Check if both are proper polynomials (at least 1D with coefficients)
        if (
            self.dim() >= 1
            and isinstance(other, ChebyshevPolynomialT)
            and other.dim() >= 1
        ):
            return chebyshev_polynomial_t_multiply(self, other)
        # Handle scalar cases: if self is 0-dim, just do tensor mul
        if self.dim() == 0:
            return self.as_subclass(Tensor) * other
        return chebyshev_polynomial_t_scale(self, other)

    def __rmul__(
        self, other: Union["ChebyshevPolynomialT", Tensor]
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_multiply import (
            chebyshev_polynomial_t_multiply,
        )
        from ._chebyshev_polynomial_t_scale import (
            chebyshev_polynomial_t_scale,
        )

        # Check if both are proper polynomials (at least 1D with coefficients)
        if (
            self.dim() >= 1
            and isinstance(other, ChebyshevPolynomialT)
            and other.dim() >= 1
        ):
            return chebyshev_polynomial_t_multiply(other, self)
        # Handle scalar cases: if self is 0-dim, just do tensor mul
        if self.dim() == 0:
            return self.as_subclass(Tensor) * other
        return chebyshev_polynomial_t_scale(self, other)

    def __pow__(self, n: int) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_pow import chebyshev_polynomial_t_pow

        return chebyshev_polynomial_t_pow(self, n)

    def __floordiv__(
        self, other: "ChebyshevPolynomialT"
    ) -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_div import chebyshev_polynomial_t_div

        return chebyshev_polynomial_t_div(self, other)

    def __mod__(self, other: "ChebyshevPolynomialT") -> "ChebyshevPolynomialT":
        from ._chebyshev_polynomial_t_mod import chebyshev_polynomial_t_mod

        return chebyshev_polynomial_t_mod(self, other)

    def __repr__(self) -> str:
        return f"ChebyshevPolynomialT({Tensor.__repr__(self)})"


def chebyshev_polynomial_t(coeffs: Tensor) -> ChebyshevPolynomialT:
    """Create Chebyshev series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of T_k(x).
        Must have at least one coefficient.

    Returns
    -------
    ChebyshevPolynomialT
        Chebyshev series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))  # 1*T_0 + 2*T_1 + 3*T_2
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Chebyshev series must have at least one coefficient"
        )

    return ChebyshevPolynomialT(coeffs)
