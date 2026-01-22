import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_mulx(
    a: HermitePolynomialH,
) -> HermitePolynomialH:
    """Multiply Physicists' Hermite series by x.

    Uses the recurrence relation:
        x * H_k(x) = H_{k+1}(x)/2 + k * H_{k-1}(x)

    Parameters
    ----------
    a : HermitePolynomialH
        Series to multiply by x.

    Returns
    -------
    HermitePolynomialH
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    The Hermite recurrence relation is:
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)

    Solving for x * H_n(x):
        x * H_n(x) = (H_{n+1}(x) + 2n * H_{n-1}(x)) / 2
                   = H_{n+1}(x)/2 + n * H_{n-1}(x)

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0]))  # H_0 = 1
    >>> b = hermite_polynomial_h_mulx(a)
    >>> b  # x * H_0 = x = H_1/2
    HermitePolynomialH(tensor([0. , 0.5]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * H_k = H_{k+1}/2 + k * H_{k-1}
    # For k=0: x * H_0 = H_1/2, contributes c_0/2 to coefficient of H_1
    result[..., 1] = result[..., 1] + coeffs[..., 0] / 2.0

    # For k >= 1: x * H_k = H_{k+1}/2 + k * H_{k-1}
    for k in range(1, n):
        c_k = coeffs[..., k]

        # Contribution to H_{k-1} coefficient: k * c_k
        result[..., k - 1] = result[..., k - 1] + k * c_k
        # Contribution to H_{k+1} coefficient: c_k / 2
        result[..., k + 1] = result[..., k + 1] + c_k / 2.0

    return hermite_polynomial_h(result)
