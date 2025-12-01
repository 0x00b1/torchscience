import torch

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def _derivative_coeffs_w(coeffs: torch.Tensor) -> torch.Tensor:
    """Compute derivative coefficients for Chebyshev W series (single step).

    For Chebyshev W polynomials, we use the recurrence:
        W_{n+1}(x) = 2x*W_n(x) - W_{n-1}(x)

    Differentiating:
        W'_{n+1}(x) = 2*W_n(x) + 2x*W'_n(x) - W'_{n-1}(x)

    We compute derivatives iteratively and express in W basis.
    """
    n = coeffs.shape[-1]

    if n <= 1:
        result_shape = list(coeffs.shape)
        result_shape[-1] = 1
        return torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

    # Build derivative polynomials for each W_k in the W basis
    # dW_0/dx = 0 -> []
    # dW_1/dx = 2 -> [2] (i.e., 2*W_0)
    # dW_k/dx for k >= 2: use recurrence

    # Result has degree n-2 (n-1 coefficients)
    result_shape = list(coeffs.shape)
    result_shape[-1] = n - 1
    d_coeffs = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # For each input coefficient c_k (k >= 1), compute contribution to derivative
    # We precompute derivatives of W_k in W basis

    # dW_k/dx coefficients stored as list of tensors
    # dW_0 = [0], dW_1 = [2], dW_2 = [-2, 4], etc.

    # Use forward recurrence to build dW_k/dx in W basis:
    # From W_{k+1} = 2x*W_k - W_{k-1}
    # W'_{k+1} = 2*W_k + 2x*W'_k - W'_{k-1}
    #
    # In terms of W basis coefficients:
    # If W'_k has coefficients [a_0, a_1, ..., a_{k-1}] (degree k-1)
    # Then 2x*W'_k contributes via the multiplication-by-x formula for W:
    # x*W_j = 0.5*(W_{j-1} + W_{j+1}) for j >= 1, with W_{-1} = -W_0
    # x*W_0 = 0.5*(W_1 - W_0)
    #
    # This is getting complex. Let me use a simpler approach:
    # Compute derivatives numerically via the power basis.

    # Simpler approach: use the relationship that works for all Chebyshev types
    # d/dx sum(c_k W_k) can be computed using backward recurrence

    deg = n - 1  # highest degree term

    # Backward recurrence for Chebyshev W derivatives
    # Similar to T/U/V but with W-specific coefficients
    # The key relationship: sum_k c_k * W'_k in W basis

    # For W polynomials, use the identity:
    # W'_n(x) = 2 * (W_{n-1} + W_{n-3} + W_{n-5} + ... + W_r)
    # where r = 0 if n is odd, r = 1 if n is even (but this needs verification)

    # Actually for W polynomials specifically:
    # dW_n/dx = 2n * sum_{k=0}^{n-1} W_k / (something)

    # Let's use a direct computation approach:
    # Compute the derivative of each W_k term and accumulate

    # Pre-compute derivatives of basis polynomials W_k for k up to deg
    # Store as coefficients in W basis

    # dW_0 = 0 (constant)
    # dW_1 = 2 (constant)
    # dW_k for k >= 2: use forward recurrence from W'_{k+1} = 2*W_k + 2x*W'_k - W'_{k-1}

    # Build derivatives iteratively
    # d_basis[k] = coefficients of dW_k/dx in W basis

    if deg == 0:
        return d_coeffs

    # d_basis[k] will hold coefficients of dW_k/dx (length k for k >= 1)
    # dW_0/dx = 0, dW_1/dx = 2*W_0

    # For efficiency, we'll accumulate directly into d_coeffs

    # Contribution from c_1 * W_1: derivative is c_1 * 2 * W_0
    if deg >= 1:
        d_coeffs[..., 0] = d_coeffs[..., 0] + coeffs[..., 1] * 2.0

    if deg >= 2:
        # For k >= 2, compute dW_k/dx using forward recurrence and accumulate
        # We need to track dW_{k-1}/dx and dW_{k-2}/dx

        # Initialize: dW_1 = [2], dW_0 = [0]
        # dW_1/dx = 2*W_0, so d_prev = [2]
        # dW_0/dx = 0, so d_prev_prev has no contribution

        # Store derivatives as padded tensors for easier manipulation
        max_deriv_deg = deg - 1  # dW_deg has degree deg-1 at most

        # d_k_minus_2[j] = coeff of W_j in dW_{k-2}/dx
        # d_k_minus_1[j] = coeff of W_j in dW_{k-1}/dx

        batch_shape = coeffs.shape[:-1]
        d_k_minus_2 = torch.zeros(
            (*batch_shape, max_deriv_deg + 1),
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        d_k_minus_1 = torch.zeros(
            (*batch_shape, max_deriv_deg + 1),
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        d_k_minus_1[..., 0] = 2.0  # dW_1/dx = 2*W_0

        for k in range(2, deg + 1):
            # Compute dW_k/dx from recurrence:
            # W_k = 2x*W_{k-1} - W_{k-2}
            # W'_k = 2*W_{k-1} + 2x*W'_{k-1} - W'_{k-2}
            #
            # In W basis: 2*W_{k-1} contributes delta at index k-1
            # 2x*W'_{k-1}: multiply W'_{k-1} by x and double
            # For W polynomials: x*W_j = 0.5*(W_{j+1} + W_{j-1}) with W_{-1} = -W_0

            d_k = torch.zeros(
                (*batch_shape, max_deriv_deg + 1),
                dtype=coeffs.dtype,
                device=coeffs.device,
            )

            # Term 1: 2*W_{k-1}
            if k - 1 <= max_deriv_deg:
                d_k[..., k - 1] = d_k[..., k - 1] + 2.0

            # Term 2: 2x * d_k_minus_1 (multiply by x each coeff, then double)
            # x * W_j = 0.5 * (W_{j+1} + W_{j-1}) for j >= 1
            # x * W_0 = 0.5 * (W_1 - W_0) [since W_{-1} = -W_0 for W polynomials]
            for j in range(max_deriv_deg + 1):
                c = d_k_minus_1[..., j]
                if j == 0:
                    # x * W_0 = 0.5 * (W_1 - W_0)
                    d_k[..., 0] = d_k[..., 0] - c  # -0.5 * 2 = -1
                    if 1 <= max_deriv_deg:
                        d_k[..., 1] = d_k[..., 1] + c  # +0.5 * 2 = +1
                else:
                    # x * W_j = 0.5 * (W_{j-1} + W_{j+1})
                    d_k[..., j - 1] = d_k[..., j - 1] + c  # 0.5 * 2 = 1
                    if j + 1 <= max_deriv_deg:
                        d_k[..., j + 1] = d_k[..., j + 1] + c

            # Term 3: -d_k_minus_2
            d_k = d_k - d_k_minus_2

            # Accumulate contribution from c_k * dW_k/dx
            c_k = coeffs[..., k]
            for j in range(min(k, n - 1)):  # dW_k has degree k-1
                d_coeffs[..., j] = d_coeffs[..., j] + c_k * d_k[..., j]

            # Shift for next iteration
            d_k_minus_2 = d_k_minus_1
            d_k_minus_1 = d_k

    return d_coeffs


def chebyshev_polynomial_w_derivative(
    a: ChebyshevPolynomialW,
    order: int = 1,
) -> ChebyshevPolynomialW:
    """Compute derivative of Chebyshev W series.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    ChebyshevPolynomialW
        Derivative series.

    Raises
    ------
    ValueError
        If order is negative.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    Uses the recurrence relation for Chebyshev W polynomials:
        W_{n+1}(x) = 2x*W_n(x) - W_{n-1}(x)

    Differentiating gives:
        W'_{n+1}(x) = 2*W_n(x) + 2x*W'_n(x) - W'_{n-1}(x)

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([0.0, 0.0, 1.0]))  # W_2
    >>> da = chebyshev_polynomial_w_derivative(a)
    >>> da.coeffs  # d/dx W_2 = -2*W_0 + 4*W_1
    tensor([-2.,  4.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return ChebyshevPolynomialW(coeffs=a.coeffs.clone())

    coeffs = a.coeffs

    # Apply derivative 'order' times
    for _ in range(order):
        coeffs = _derivative_coeffs_w(coeffs)

    return ChebyshevPolynomialW(coeffs=coeffs)
