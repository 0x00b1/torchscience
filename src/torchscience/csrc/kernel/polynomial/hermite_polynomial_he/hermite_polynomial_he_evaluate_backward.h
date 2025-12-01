#pragma once

#include <cstdint>
#include <vector>

namespace torchscience::kernel::polynomial {

/**
 * Backward pass for Probabilists' Hermite polynomial evaluation.
 *
 * For grad_coeffs:
 *   d(output)/d(c_k) = He_k(x)
 *   So grad_coeffs[k] = grad_output * He_k(x)
 *
 * For grad_x:
 *   He_k'(x) = k * He_{k-1}(x)  for k >= 1, He_0'(x) = 0
 *   df/dx = sum_k c_k * He_k'(x)
 *         = sum_{k=1}^{N-1} c_k * k * He_{k-1}(x)
 *         = sum_{j=0}^{N-2} (j+1) * c_{j+1} * He_j(x)
 *
 * @param grad_coeffs Output: gradient w.r.t. coefficients (size N)
 * @param grad_output Upstream gradient (scalar for this point)
 * @param coeffs Polynomial coefficients (size N)
 * @param x Evaluation point
 * @param N Number of coefficients
 * @return Gradient with respect to x
 */
template <typename T>
T hermite_polynomial_he_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute all Hermite He polynomial values He_0(x), He_1(x), ..., He_{N-1}(x)
    std::vector<T> He(N);
    He[0] = T(1);  // He_0(x) = 1

    if (N > 1) {
        He[1] = x;  // He_1(x) = x
    }

    // Forward recurrence: He_{k+1}(x) = x * He_k(x) - k * He_{k-1}(x)
    for (int64_t k = 1; k < N - 1; ++k) {
        He[k + 1] = x * He[k] - T(k) * He[k - 1];
    }

    // Compute grad_coeffs: grad_coeffs[k] = grad_output * He_k(x)
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output * He[k];
    }

    // Compute grad_x:
    // df/dx = sum_{k=1}^{N-1} c_k * k * He_{k-1}(x)
    //       = sum_{j=0}^{N-2} (j+1) * c_{j+1} * He_j(x)
    T grad_x = T(0);
    for (int64_t j = 0; j < N - 1; ++j) {
        grad_x += T(j + 1) * coeffs[j + 1] * He[j];
    }
    grad_x *= grad_output;

    return grad_x;
}

}  // namespace torchscience::kernel::polynomial
