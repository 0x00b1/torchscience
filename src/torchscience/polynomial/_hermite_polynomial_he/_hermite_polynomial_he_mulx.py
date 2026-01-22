import torch

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)


def hermite_polynomial_he_mulx(
    a: HermitePolynomialHe,
) -> HermitePolynomialHe:
    """Multiply Probabilists' Hermite series by x.

    Uses the recurrence relation:
        x * He_k(x) = He_{k+1}(x) + k * He_{k-1}(x)

    Parameters
    ----------
    a : HermitePolynomialHe
        Series to multiply by x.

    Returns
    -------
    HermitePolynomialHe
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    The Hermite_e recurrence relation is:
        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

    Solving for x * He_n(x):
        x * He_n(x) = He_{n+1}(x) + n * He_{n-1}(x)

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0]))  # He_0 = 1
    >>> b = hermite_polynomial_he_mulx(a)
    >>> b  # x * He_0 = x = He_1
    HermitePolynomialHe(tensor([0., 1.]))
    """
    # Convert to plain tensor for operations
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * He_k = He_{k+1} + k * He_{k-1}
    # For k=0: x * He_0 = He_1, contributes c_0 to coefficient of He_1
    result[..., 1] = result[..., 1] + coeffs[..., 0]

    # For k >= 1: x * He_k = He_{k+1} + k * He_{k-1}
    for k in range(1, n):
        c_k = coeffs[..., k]

        # Contribution to He_{k-1} coefficient: k * c_k
        result[..., k - 1] = result[..., k - 1] + k * c_k
        # Contribution to He_{k+1} coefficient: c_k
        result[..., k + 1] = result[..., k + 1] + c_k

    return hermite_polynomial_he(result)
