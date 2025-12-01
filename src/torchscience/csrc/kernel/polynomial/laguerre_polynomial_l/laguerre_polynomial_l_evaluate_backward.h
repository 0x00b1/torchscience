#pragma once

#include <cstdint>
#include <vector>

namespace torchscience::kernel::polynomial {

/**
 * Backward pass for Laguerre polynomial evaluation.
 *
 * Computes gradients with respect to coefficients and x.
 *
 * For grad_coeffs:
 *   d(output)/d(c_k) = L_k(x)
 *   So grad_coeffs[k] = grad_output * L_k(x)
 *
 * For grad_x:
 *   L_k'(x) = -sum_{j=0}^{k-1} L_j(x)  for k >= 1, L_0'(x) = 0
 *   df/dx = sum_k c_k * L_k'(x)
 *         = -sum_k c_k * sum_{j=0}^{k-1} L_j(x)
 *         = -sum_j L_j(x) * sum_{k>j} c_k
 *
 * @param grad_coeffs Output: gradient w.r.t. coefficients (size N)
 * @param grad_output Upstream gradient (scalar for this point)
 * @param coeffs Polynomial coefficients (size N)
 * @param x Evaluation point
 * @param N Number of coefficients
 * @return Gradient with respect to x
 */
template <typename T>
T laguerre_polynomial_l_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute all Laguerre polynomial values L_0(x), L_1(x), ..., L_{N-1}(x)
    std::vector<T> L(N);
    L[0] = T(1);  // L_0(x) = 1

    if (N > 1) {
        L[1] = T(1) - x;  // L_1(x) = 1 - x
    }

    // Forward recurrence: L_{k+1}(x) = ((2k+1-x) * L_k(x) - k * L_{k-1}(x)) / (k+1)
    for (int64_t k = 1; k < N - 1; ++k) {
        L[k + 1] = ((T(2 * k + 1) - x) * L[k] - T(k) * L[k - 1]) / T(k + 1);
    }

    // Compute grad_coeffs: grad_coeffs[k] = grad_output * L_k(x)
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output * L[k];
    }

    // Compute grad_x:
    // df/dx = sum_k c_k * L_k'(x) where L_k'(x) = -sum_{j=0}^{k-1} L_j(x)
    // = -sum_j L_j(x) * sum_{k=j+1}^{N-1} c_k
    // Let suffix_sum[j] = sum_{k=j+1}^{N-1} c_k
    // df/dx = -sum_j L_j(x) * suffix_sum[j]

    // Compute suffix sums: suffix_sum[j] = c_{j+1} + c_{j+2} + ... + c_{N-1}
    std::vector<T> suffix_sum(N);
    suffix_sum[N - 1] = T(0);  // No coefficients above N-1
    for (int64_t j = N - 2; j >= 0; --j) {
        suffix_sum[j] = suffix_sum[j + 1] + coeffs[j + 1];
    }

    T grad_x = T(0);
    for (int64_t j = 0; j < N; ++j) {
        grad_x -= L[j] * suffix_sum[j];
    }
    grad_x *= grad_output;

    return grad_x;
}

}  // namespace torchscience::kernel::polynomial
