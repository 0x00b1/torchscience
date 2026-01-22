import torch
from torch import Tensor

from .._parameter_error import ParameterError
from .._polynomial_error import PolynomialError

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


class JacobiPolynomialP(Tensor):
    """Jacobi polynomial series P_n^{(alpha, beta)}(x).

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * P_k^{(alpha,beta)}(x)

    where P_k^{(alpha,beta)}(x) are Jacobi polynomials with parameters alpha and beta.

    The tensor itself stores the coefficients in ascending order, shape (..., N)
    where N = degree + 1. coeffs[..., k] is the coefficient of P_k^{(alpha,beta)}(x).
    Batch dimensions come first, coefficient dimension last.

    Attributes
    ----------
    alpha : Tensor
        Parameter alpha, must be > -1. Scalar tensor for consistency.
    beta : Tensor
        Parameter beta, must be > -1. Scalar tensor for consistency.

    Notes
    -----
    The standard domain for Jacobi polynomials is [-1, 1].

    The Jacobi polynomials are orthogonal on [-1, 1] with weight function
    w(x) = (1-x)^alpha * (1+x)^beta.

    Special cases:
        - alpha = beta = 0: Legendre polynomials P_n(x)
        - alpha = beta = -1/2: Chebyshev polynomials of the first kind T_n(x)
        - alpha = beta = 1/2: Chebyshev polynomials of the second kind U_n(x)
        - alpha = beta: Gegenbauer (ultraspherical) polynomials
    """

    DOMAIN = (-1.0, 1.0)
    alpha: Tensor
    beta: Tensor

    @staticmethod
    def __new__(
        cls, data, alpha: Tensor, beta: Tensor, *, dtype=None, device=None
    ):
        if isinstance(data, Tensor):
            tensor = data.detach().clone()
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            if device is not None:
                tensor = tensor.to(device=device)
        else:
            tensor = torch.as_tensor(data, dtype=dtype, device=device)
        instance = tensor.as_subclass(cls)
        instance.alpha = alpha
        instance.beta = beta
        return instance

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = super().__torch_function__(func, types, args, kwargs)

        # Propagate alpha, beta to result for shape-preserving ops
        if func in _SHAPE_PRESERVING_OPS | _POLYNOMIAL_RETURNING_OPS:
            if isinstance(result, Tensor) and not isinstance(result, cls):
                # Find source to copy params from
                for arg in args:
                    if isinstance(arg, cls):
                        result = result.as_subclass(cls)
                        result.alpha = arg.alpha
                        result.beta = arg.beta
                        break

        return result

    def __call__(self, x: Tensor) -> Tensor:
        from ._jacobi_polynomial_p_evaluate import jacobi_polynomial_p_evaluate

        return jacobi_polynomial_p_evaluate(self, x)

    def __add__(self, other: "JacobiPolynomialP") -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_add import jacobi_polynomial_p_add

        return jacobi_polynomial_p_add(self, other)

    def __radd__(self, other: "JacobiPolynomialP") -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_add import jacobi_polynomial_p_add

        return jacobi_polynomial_p_add(other, self)

    def __sub__(self, other: "JacobiPolynomialP") -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_subtract import jacobi_polynomial_p_subtract

        return jacobi_polynomial_p_subtract(self, other)

    def __rsub__(self, other: "JacobiPolynomialP") -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_subtract import jacobi_polynomial_p_subtract

        return jacobi_polynomial_p_subtract(other, self)

    def __neg__(self) -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_negate import jacobi_polynomial_p_negate

        return jacobi_polynomial_p_negate(self)

    def __mul__(self, other):
        if isinstance(other, JacobiPolynomialP):
            from ._jacobi_polynomial_p_multiply import (
                jacobi_polynomial_p_multiply,
            )

            return jacobi_polynomial_p_multiply(self, other)
        if isinstance(other, Tensor):
            from ._jacobi_polynomial_p_scale import jacobi_polynomial_p_scale

            return jacobi_polynomial_p_scale(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, JacobiPolynomialP):
            from ._jacobi_polynomial_p_multiply import (
                jacobi_polynomial_p_multiply,
            )

            return jacobi_polynomial_p_multiply(other, self)
        if isinstance(other, Tensor):
            from ._jacobi_polynomial_p_scale import jacobi_polynomial_p_scale

            return jacobi_polynomial_p_scale(self, other)
        return NotImplemented

    def __pow__(self, n: int) -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_pow import jacobi_polynomial_p_pow

        return jacobi_polynomial_p_pow(self, n)

    def __floordiv__(self, other: "JacobiPolynomialP") -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_div import jacobi_polynomial_p_div

        return jacobi_polynomial_p_div(self, other)

    def __mod__(self, other: "JacobiPolynomialP") -> "JacobiPolynomialP":
        from ._jacobi_polynomial_p_mod import jacobi_polynomial_p_mod

        return jacobi_polynomial_p_mod(self, other)

    def __repr__(self) -> str:
        return f"JacobiPolynomialP({Tensor.__repr__(self)}, alpha={self.alpha}, beta={self.beta})"


def jacobi_polynomial_p(
    coeffs: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> JacobiPolynomialP:
    """Create Jacobi series from coefficient tensor and parameters.

    Parameters
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N).
        coeffs[..., k] is the coefficient of P_k^{(alpha,beta)}(x).
        Must have at least one coefficient.
    alpha : Tensor or float
        Parameter alpha, must be > -1.
    beta : Tensor or float
        Parameter beta, must be > -1.

    Returns
    -------
    JacobiPolynomialP
        Jacobi series instance.

    Raises
    ------
    PolynomialError
        If coeffs is empty (size 0 in last dimension).
    ParameterError
        If alpha <= -1 or beta <= -1.

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.5, beta=0.5)
    >>> c[0]
    tensor(1.)
    >>> c.alpha
    tensor(0.5)
    """
    if coeffs.numel() == 0 or coeffs.shape[-1] == 0:
        raise PolynomialError(
            "Jacobi series must have at least one coefficient"
        )

    # Convert alpha and beta to tensors if needed
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(alpha, dtype=coeffs.dtype, device=coeffs.device)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=coeffs.dtype, device=coeffs.device)

    if (alpha <= -1).any():
        raise ParameterError(f"alpha must be > -1, got {alpha}")
    if (beta <= -1).any():
        raise ParameterError(f"beta must be > -1, got {beta}")

    return JacobiPolynomialP(coeffs, alpha, beta)
