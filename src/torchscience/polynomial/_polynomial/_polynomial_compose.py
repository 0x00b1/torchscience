from ._polynomial import Polynomial, polynomial
from ._polynomial_multiply import polynomial_multiply


def polynomial_compose(p: Polynomial, q: Polynomial) -> Polynomial:
    """Compose polynomials: compute p(q(x)).

    Uses Horner's method for efficient evaluation.

    Parameters
    ----------
    p : Polynomial
        Outer polynomial.
    q : Polynomial
        Inner polynomial (substituted for x).

    Returns
    -------
    Polynomial
        The composition p(q(x)).

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> q = polynomial(torch.tensor([1.0, 1.0]))  # x + 1
    >>> polynomial_compose(p, q)  # (x+1)^2 = x^2 + 2x + 1
    Polynomial(tensor([1., 2., 1.]))
    """
    # p IS the coefficient tensor now
    n = p.shape[-1]

    if n == 0:
        return p

    # Horner's method: p(q) = a_0 + q*(a_1 + q*(a_2 + ...))
    # Start with leading coefficient as constant polynomial
    result_coeffs = p[..., n - 1 : n]  # Keep dimension
    result = polynomial(result_coeffs)

    for i in range(n - 2, -1, -1):
        # result = result * q + a_i
        result = polynomial_multiply(result, q)

        # Add coefficient a_i to the constant term of result
        # This avoids polynomial_add which has issues with batched different-degree polys
        a_i = p[..., i]  # Shape: batch_dims or scalar
        result_coeffs = result.clone()
        result_coeffs[..., 0] = result_coeffs[..., 0] + a_i
        result = polynomial(result_coeffs)

    return result
