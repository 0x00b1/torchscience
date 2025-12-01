import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from .._parameter_error import ParameterError
from .._polynomial_error import PolynomialError


@tensorclass
class JacobiPolynomialP:
    """Jacobi polynomial series.

    Represents f(x) = sum_{k=0}^{n} coeffs[..., k] * P_k^{(α,β)}(x)

    where P_k^{(α,β)}(x) are Jacobi polynomials with parameters α and β.

    Attributes
    ----------
    coeffs : Tensor
        Coefficients in ascending order, shape (..., N) where N = degree + 1.
        coeffs[..., k] is the coefficient of P_k^{(α,β)}(x).
        Batch dimensions come first, coefficient dimension last.
    alpha : Tensor
        Parameter α, must be > -1. Scalar tensor for consistency.
    beta : Tensor
        Parameter β, must be > -1. Scalar tensor for consistency.

    Notes
    -----
    The standard domain for Jacobi polynomials is [-1, 1].

    The Jacobi polynomials are orthogonal on [-1, 1] with weight function
    w(x) = (1-x)^α * (1+x)^β.

    Special cases:
        - α = β = 0: Legendre polynomials P_n(x)
        - α = β = -1/2: Chebyshev polynomials of the first kind T_n(x)
        - α = β = 1/2: Chebyshev polynomials of the second kind U_n(x)
        - α = β: Gegenbauer (ultraspherical) polynomials
    """

    coeffs: Tensor
    alpha: Tensor
    beta: Tensor

    DOMAIN = (-1.0, 1.0)

    def __post_init__(self):
        if (self.alpha <= -1).any():
            raise ParameterError(f"alpha must be > -1, got {self.alpha}")
        if (self.beta <= -1).any():
            raise ParameterError(f"beta must be > -1, got {self.beta}")

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
        coeffs[..., k] is the coefficient of P_k^{(α,β)}(x).
        Must have at least one coefficient.
    alpha : Tensor or float
        Parameter α, must be > -1.
    beta : Tensor or float
        Parameter β, must be > -1.

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
    >>> c.coeffs
    tensor([1., 2., 3.])
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

    return JacobiPolynomialP(coeffs=coeffs, alpha=alpha, beta=beta)
