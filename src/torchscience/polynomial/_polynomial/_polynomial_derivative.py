import torch

from ._polynomial import Polynomial, polynomial


def polynomial_derivative(p: Polynomial, order: int = 1) -> Polynomial:
    """Compute derivative of polynomial.

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    order : int
        Derivative order (default 1).

    Returns
    -------
    Polynomial
        Derivative d^n p / dx^n. Constant polynomial returns [0.0].

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> polynomial_derivative(p)  # 2 + 6x
    Polynomial(tensor([2., 6.]))
    """
    # p IS the coefficient tensor now
    coeffs = p

    for _ in range(order):
        n = coeffs.shape[-1]
        if n <= 1:
            # Derivative of constant is zero
            return polynomial(
                torch.zeros(
                    *coeffs.shape[:-1],
                    1,
                    dtype=coeffs.dtype,
                    device=coeffs.device,
                )
            )

        # d/dx (a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n)
        # = a_1 + 2*a_2*x + 3*a_3*x^2 + ... + n*a_n*x^(n-1)
        # new_coeffs[i] = (i+1) * old_coeffs[i+1]
        indices = torch.arange(1, n, device=coeffs.device, dtype=coeffs.dtype)
        new_coeffs = coeffs[..., 1:] * indices

        coeffs = new_coeffs

    return polynomial(coeffs)
