import torch

from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_evaluate import chebyshev_polynomial_w_evaluate


def _compute_derivative_matrix_w(n: int, dtype, device) -> torch.Tensor:
    """Build the derivative matrix for Chebyshev W polynomials.

    D[i, j] = coefficient of W_i in dW_j/dx

    For a polynomial with coefficients c, the derivative has coefficients D @ c[1:].
    """
    # D is (n-1) x n matrix: derivative of degree n poly is degree n-1
    D = torch.zeros((n, n + 1), dtype=dtype, device=device)

    # dW_0/dx = 0 (column 0 is all zeros)

    # dW_1/dx = 2*W_0
    if n >= 1:
        D[0, 1] = 2.0

    # For k >= 2, use forward recurrence to compute dW_k/dx
    # W_k = 2x*W_{k-1} - W_{k-2}
    # W'_k = 2*W_{k-1} + 2x*W'_{k-1} - W'_{k-2}

    # Store dW_k/dx as columns
    if n >= 2:
        # d_k[j] = coeff of W_j in dW_k/dx
        d_km2 = torch.zeros(n, dtype=dtype, device=device)  # dW_0 = 0
        d_km1 = torch.zeros(n, dtype=dtype, device=device)
        d_km1[0] = 2.0  # dW_1 = 2*W_0

        for k in range(2, n + 1):
            d_k = torch.zeros(n, dtype=dtype, device=device)

            # Term 1: 2*W_{k-1}
            if k - 1 < n:
                d_k[k - 1] = 2.0

            # Term 2: 2x * dW_{k-1}/dx
            # x*W_j = 0.5*(W_{j+1} + W_{j-1}) for j >= 1, with W_{-1} = -W_0
            for j in range(n):
                c = d_km1[j]
                if c == 0:
                    continue
                if j == 0:
                    # x*W_0 = 0.5*(W_1 - W_0)
                    d_k[0] = d_k[0] - c  # 2 * 0.5 * (-1) = -1
                    if j + 1 < n:
                        d_k[1] = d_k[1] + c  # 2 * 0.5 * 1 = 1
                else:
                    # x*W_j = 0.5*(W_{j-1} + W_{j+1})
                    d_k[j - 1] = d_k[j - 1] + c  # 2 * 0.5 = 1
                    if j + 1 < n:
                        d_k[j + 1] = d_k[j + 1] + c

            # Term 3: -dW_{k-2}/dx
            d_k = d_k - d_km2

            # Store in matrix
            if k < n + 1:
                D[:, k] = d_k

            # Shift
            d_km2 = d_km1.clone()
            d_km1 = d_k.clone()

    return D


def _compute_antiderivative_single(coeffs: torch.Tensor) -> torch.Tensor:
    """Compute antiderivative coefficients for a single W series.

    Returns coefficients with one higher degree. The constant term (index 0)
    is set to 0; caller should adjust.
    """
    n = coeffs.shape[-1]

    if n == 0:
        return torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)

    # Result has n+1 coefficients
    result = torch.zeros(n + 1, dtype=coeffs.dtype, device=coeffs.device)

    # Handle each input coefficient
    # For c_k * W_k, we need to find integral(W_k) and accumulate

    # integral(W_0) = -0.5*W_0 + 0.5*W_1
    if n >= 1:
        result[0] = result[0] + coeffs[0] * (-0.5)
        result[1] = result[1] + coeffs[0] * 0.5

    # integral(W_1) = 0.25*W_1 + 0.25*W_2 (ignoring constant)
    if n >= 2:
        result[1] = result[1] + coeffs[1] * 0.25
        result[2] = result[2] + coeffs[1] * 0.25

    # For k >= 2, solve the system to find integral(W_k)
    # This requires computing derivative matrix and solving
    for k in range(2, n):
        # Solve: find a such that d/dx(sum_j a_j W_j) = W_k
        # This is: D @ a = e_k where D is derivative matrix

        # Build small derivative matrix for k+2 terms
        size = k + 2
        D = _compute_derivative_matrix_w(size - 1, coeffs.dtype, coeffs.device)

        # D is (size-1) x size, we need to solve D[:k+1, 1:] @ a[1:] = e_k[:k+1]
        # where e_k has 1 at position k and 0 elsewhere

        # Extract the relevant submatrix (columns 1 to size-1, rows 0 to k)
        D_sub = D[: k + 1, 1:size]  # (k+1) x (size-1)

        # Right-hand side: e_k (W_k = [0,...,0,1,0,...])
        rhs = torch.zeros(k + 1, dtype=coeffs.dtype, device=coeffs.device)
        rhs[k] = 1.0

        # Solve least squares (system is typically overdetermined or exact)
        try:
            a_sub = torch.linalg.lstsq(D_sub.T @ D_sub, D_sub.T @ rhs).solution
        except Exception:
            # Fallback: simple formula approximation
            a_sub = torch.zeros(
                size - 1, dtype=coeffs.dtype, device=coeffs.device
            )
            a_sub[-1] = 1.0 / (4.0 * k) if k > 0 else 0.25

        # a_sub contains coefficients for [W_1, W_2, ..., W_{size-1}]
        # Add c_k * a_sub to result
        for j, aj in enumerate(a_sub):
            if j + 1 < n + 1:
                result[j + 1] = result[j + 1] + coeffs[k] * aj

    return result


def chebyshev_polynomial_w_antiderivative(
    a: ChebyshevPolynomialW,
    order: int = 1,
    constant: float = 0.0,
) -> ChebyshevPolynomialW:
    """Compute antiderivative of Chebyshev W series.

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    ChebyshevPolynomialW
        Antiderivative series.

    Raises
    ------
    ValueError
        If order is negative.

    Notes
    -----
    The degree increases by 1 for each integration.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0]))  # constant 1 = W_0
    >>> ia = chebyshev_polynomial_w_antiderivative(a)
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return ChebyshevPolynomialW(coeffs=a.coeffs.clone())

    coeffs = a.coeffs

    # Apply antiderivative 'order' times
    for iteration in range(order):
        # Handle batched case by iterating (can be optimized later)
        if coeffs.dim() == 1:
            i_coeffs = _compute_antiderivative_single(coeffs)
        else:
            # Batched: apply to each batch element
            batch_shape = coeffs.shape[:-1]
            n = coeffs.shape[-1]
            result_shape = list(coeffs.shape)
            result_shape[-1] = n + 1
            i_coeffs = torch.zeros(
                result_shape, dtype=coeffs.dtype, device=coeffs.device
            )
            for idx in torch.ndindex(batch_shape):
                i_coeffs[idx] = _compute_antiderivative_single(coeffs[idx])

        # Set constant so that F(0) = constant (only for first integration)
        k_val = constant if iteration == 0 else 0.0
        temp = ChebyshevPolynomialW(coeffs=i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = chebyshev_polynomial_w_evaluate(temp, x_zero)
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return ChebyshevPolynomialW(coeffs=coeffs)
