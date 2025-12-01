#pragma once

#include <cstdint>
#include <vector>

namespace torchscience::kernel::polynomial {

/**
 * Backward pass for Physicists' Hermite polynomial evaluation.
 *
 * For grad_coeffs:
 *   d(output)/d(c_k) = H_k(x)
 *   So grad_coeffs[k] = grad_output * H_k(x)
 *
 * For grad_x:
 *   H_k'(x) = 2k * H_{k-1}(x)  for k >= 1, H_0'(x) = 0
 *   df/dx = sum_k c_k * H_k'(x)
 *         = sum_{k=1}^{N-1} c_k * 2k * H_{k-1}(x)
 *         = sum_{j=0}^{N-2} 2(j+1) * c_{j+1} * H_j(x)
 *
 * @param grad_coeffs Output: gradient w.r.t. coefficients (size N)
 * @param grad_output Upstream gradient (scalar for this point)
 * @param coeffs Polynomial coefficients (size N)
 * @param x Evaluation point
 * @param N Number of coefficients
 * @return Gradient with respect to x
 */
template <typename T>
T hermite_polynomial_h_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute all Hermite polynomial values H_0(x), H_1(x), ..., H_{N-1}(x)
    std::vector<T> H(N);
    H[0] = T(1);  // H_0(x) = 1

    if (N > 1) {
        H[1] = T(2) * x;  // H_1(x) = 2x
    }

    // Forward recurrence: H_{k+1}(x) = 2x * H_k(x) - 2k * H_{k-1}(x)
    for (int64_t k = 1; k < N - 1; ++k) {
        H[k + 1] = T(2) * x * H[k] - T(2 * k) * H[k - 1];
    }

    // Compute grad_coeffs: grad_coeffs[k] = grad_output * H_k(x)
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output * H[k];
    }

    // Compute grad_x:
    // df/dx = sum_{k=1}^{N-1} c_k * 2k * H_{k-1}(x)
    //       = sum_{j=0}^{N-2} 2(j+1) * c_{j+1} * H_j(x)
    T grad_x = T(0);
    for (int64_t j = 0; j < N - 1; ++j) {
        grad_x += T(2 * (j + 1)) * coeffs[j + 1] * H[j];
    }
    grad_x *= grad_output;

    return grad_x;
}

}  // namespace torchscience::kernel::polynomial
