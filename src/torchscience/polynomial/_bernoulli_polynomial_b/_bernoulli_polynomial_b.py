"""Bernoulli polynomial series B_n(x)."""

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


class BernoulliPolynomialB(Tensor):
    r"""Bernoulli polynomial series.

    Represents f(x) = sum_{k=0}^{n} c[k] * B_k(x)

    where B_k(x) are Bernoulli polynomials defined by:

    .. math::

        B_n(x) = \sum_{k=0}^{n} \binom{n}{k} B_k x^{n-k}

    where B_k are the Bernoulli numbers.

    Shape: (...batch, N) where N = degree + 1
    c[..., k] is the coefficient of B_k(x).

    Notes
    -----
    The Bernoulli polynomials have important properties:
    - B'_n(x) = n * B_{n-1}(x)  (derivative)
    - B_n(x+1) - B_n(x) = n * x^{n-1}  (difference property)
    - B_n(1-x) = (-1)^n * B_n(x)  (reflection property)

    The domain is all real numbers (no natural domain restriction).
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

    def __call__(self, x: Tensor) -> Tensor:
        from ._bernoulli_polynomial_b_evaluate import (
            bernoulli_polynomial_b_evaluate,
        )

        return bernoulli_polynomial_b_evaluate(self, x)

    def __add__(self, other: "BernoulliPolynomialB") -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_add import bernoulli_polynomial_b_add

        return bernoulli_polynomial_b_add(self, other)

    def __radd__(
        self, other: "BernoulliPolynomialB"
    ) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_add import bernoulli_polynomial_b_add

        return bernoulli_polynomial_b_add(other, self)

    def __sub__(self, other: "BernoulliPolynomialB") -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_subtract import (
            bernoulli_polynomial_b_subtract,
        )

        return bernoulli_polynomial_b_subtract(self, other)

    def __rsub__(
        self, other: "BernoulliPolynomialB"
    ) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_subtract import (
            bernoulli_polynomial_b_subtract,
        )

        return bernoulli_polynomial_b_subtract(other, self)

    def __neg__(self) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_negate import (
            bernoulli_polynomial_b_negate,
        )

        return bernoulli_polynomial_b_negate(self)

    def __mul__(
        self, other: Union["BernoulliPolynomialB", Tensor]
    ) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_multiply import (
            bernoulli_polynomial_b_multiply,
        )
        from ._bernoulli_polynomial_b_scale import (
            bernoulli_polynomial_b_scale,
        )

        if isinstance(other, BernoulliPolynomialB):
            return bernoulli_polynomial_b_multiply(self, other)
        return bernoulli_polynomial_b_scale(self, other)

    def __rmul__(
        self, other: Union["BernoulliPolynomialB", Tensor]
    ) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_multiply import (
            bernoulli_polynomial_b_multiply,
        )
        from ._bernoulli_polynomial_b_scale import (
            bernoulli_polynomial_b_scale,
        )

        if isinstance(other, BernoulliPolynomialB):
            return bernoulli_polynomial_b_multiply(other, self)
        return bernoulli_polynomial_b_scale(self, other)

    def __pow__(self, n: int) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_pow import bernoulli_polynomial_b_pow

        return bernoulli_polynomial_b_pow(self, n)

    def __floordiv__(
        self, other: "BernoulliPolynomialB"
    ) -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_div import bernoulli_polynomial_b_div

        return bernoulli_polynomial_b_div(self, other)

    def __mod__(self, other: "BernoulliPolynomialB") -> "BernoulliPolynomialB":
        from ._bernoulli_polynomial_b_mod import bernoulli_polynomial_b_mod

        return bernoulli_polynomial_b_mod(self, other)

    def __repr__(self) -> str:
        return f"BernoulliPolynomialB({Tensor.__repr__(self)})"


def bernoulli_polynomial_b(coeffs: Tensor) -> BernoulliPolynomialB:
    """Create Bernoulli polynomial series from coefficient tensor.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of B_k(x).
        Must have at least one coefficient.

    Returns
    -------
    BernoulliPolynomialB
        Bernoulli polynomial series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).

    Examples
    --------
    >>> c = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 3.0]))
    >>> # Represents 1*B_0(x) + 2*B_1(x) + 3*B_2(x)
    >>> c[0]
    tensor(1.)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Bernoulli polynomial series must have at least one coefficient"
        )

    return BernoulliPolynomialB(coeffs)
