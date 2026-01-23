"""Euler polynomial series E_n(x)."""

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


class EulerPolynomialE(Tensor):
    r"""Euler polynomial series.

    Represents f(x) = sum_{k=0}^{n} c[k] * E_k(x)

    where E_k(x) are Euler polynomials defined by:

    .. math::

        E_n(x) = \sum_{k=0}^{n} \binom{n}{k} \frac{E_k}{2^k} \left(x - \frac{1}{2}\right)^{n-k}

    where E_k are the Euler numbers.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of E_k(x).

    Notes
    -----
    The Euler polynomials have important properties:
    - E'_n(x) = n * E_{n-1}(x)  (derivative)
    - E_n(1-x) = (-1)^n * E_n(x)  (reflection property)
    - E_n(x+1) + E_n(x) = 2x^n  (addition property)

    The domain is all real numbers (no natural domain restriction).
    """

    DOMAIN = (-float("inf"), float("inf"))

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
        from ._euler_polynomial_e_evaluate import (
            euler_polynomial_e_evaluate,
        )

        return euler_polynomial_e_evaluate(self, x)

    def __add__(self, other: "EulerPolynomialE") -> "EulerPolynomialE":
        from ._euler_polynomial_e_add import euler_polynomial_e_add

        return euler_polynomial_e_add(self, other)

    def __radd__(self, other: "EulerPolynomialE") -> "EulerPolynomialE":
        from ._euler_polynomial_e_add import euler_polynomial_e_add

        return euler_polynomial_e_add(other, self)

    def __sub__(self, other: "EulerPolynomialE") -> "EulerPolynomialE":
        from ._euler_polynomial_e_subtract import (
            euler_polynomial_e_subtract,
        )

        return euler_polynomial_e_subtract(self, other)

    def __rsub__(self, other: "EulerPolynomialE") -> "EulerPolynomialE":
        from ._euler_polynomial_e_subtract import (
            euler_polynomial_e_subtract,
        )

        return euler_polynomial_e_subtract(other, self)

    def __neg__(self) -> "EulerPolynomialE":
        from ._euler_polynomial_e_negate import (
            euler_polynomial_e_negate,
        )

        return euler_polynomial_e_negate(self)

    def __mul__(
        self, other: Union["EulerPolynomialE", Tensor]
    ) -> "EulerPolynomialE":
        from ._euler_polynomial_e_multiply import (
            euler_polynomial_e_multiply,
        )
        from ._euler_polynomial_e_scale import (
            euler_polynomial_e_scale,
        )

        if isinstance(other, EulerPolynomialE):
            return euler_polynomial_e_multiply(self, other)
        return euler_polynomial_e_scale(self, other)

    def __rmul__(
        self, other: Union["EulerPolynomialE", Tensor]
    ) -> "EulerPolynomialE":
        from ._euler_polynomial_e_multiply import (
            euler_polynomial_e_multiply,
        )
        from ._euler_polynomial_e_scale import (
            euler_polynomial_e_scale,
        )

        if isinstance(other, EulerPolynomialE):
            return euler_polynomial_e_multiply(other, self)
        return euler_polynomial_e_scale(self, other)

    def __pow__(self, n: int) -> "EulerPolynomialE":
        from ._euler_polynomial_e_pow import euler_polynomial_e_pow

        return euler_polynomial_e_pow(self, n)

    def __floordiv__(self, other: "EulerPolynomialE") -> "EulerPolynomialE":
        from ._euler_polynomial_e_div import euler_polynomial_e_div

        return euler_polynomial_e_div(self, other)

    def __mod__(self, other: "EulerPolynomialE") -> "EulerPolynomialE":
        from ._euler_polynomial_e_mod import euler_polynomial_e_mod

        return euler_polynomial_e_mod(self, other)

    def __repr__(self) -> str:
        return f"EulerPolynomialE({Tensor.__repr__(self)})"


def euler_polynomial_e(coeffs: Tensor) -> EulerPolynomialE:
    """Create Euler polynomial series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of E_k(x).
        Must have at least one coefficient.

    Returns
    -------
    EulerPolynomialE
        Euler polynomial series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = euler_polynomial_e(torch.tensor([1.0, 2.0, 3.0]))
    >>> # Represents 1*E_0(x) + 2*E_1(x) + 3*E_2(x)
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Euler polynomial series must have at least one coefficient"
        )

    return EulerPolynomialE(coeffs)
