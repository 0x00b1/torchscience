#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace torchscience::kernel::polynomial {

/**
 * Backward pass for Gegenbauer polynomial evaluation.
 *
 * Computes gradients with respect to coefficients, x, and lambda.
 *
 * For grad_coeffs:
 *   d(output)/d(c_k) = C_k^{λ}(x)
 *   So grad_coeffs[k] = grad_output * C_k^{λ}(x)
 *
 * For grad_x:
 *   dC_n^{λ}(x)/dx = 2λ * C_{n-1}^{λ+1}(x) for n >= 1
 *   df/dx = sum_{k>=1} c_k * 2λ * C_{k-1}^{λ+1}(x)
 *
 * For grad_lambda:
 *   We need dC_k^{λ}(x)/dλ which requires forward computation.
 *
 * @param grad_coeffs Output: gradient w.r.t. coefficients (size N)
 * @param grad_output Upstream gradient (scalar for this point)
 * @param coeffs Polynomial coefficients (size N)
 * @param x Evaluation point
 * @param lambda_ The λ parameter
 * @param N Number of coefficients
 * @return Tuple of (grad_x, grad_lambda)
 */
template <typename T>
std::tuple<T, T> gegenbauer_polynomial_c_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    T lambda_,
    int64_t N
) {
    if (N == 0) {
        return {T(0), T(0)};
    }

    // Compute all Gegenbauer polynomial values C_k^{λ}(x) for k = 0..N-1
    std::vector<T> C(N);
    C[0] = T(1);  // C_0^{λ}(x) = 1

    if (N > 1) {
        C[1] = T(2) * lambda_ * x;  // C_1^{λ}(x) = 2λx
    }

    // Forward recurrence: C_{k+1}(x) = A_k * x * C_k(x) - B_k * C_{k-1}(x)
    // where A_k = 2(k+λ)/(k+1), B_k = (k+2λ-1)/(k+1)
    for (int64_t k = 1; k < N - 1; ++k) {
        T a_k = T(2) * (T(k) + lambda_) / T(k + 1);
        T b_k = (T(k) + T(2) * lambda_ - T(1)) / T(k + 1);
        C[k + 1] = a_k * x * C[k] - b_k * C[k - 1];
    }

    // Compute grad_coeffs: grad_coeffs[k] = grad_output * C_k^{λ}(x)
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output * C[k];
    }

    // Compute grad_x using C_k^{λ+1}(x) polynomials
    // dC_n^{λ}(x)/dx = 2λ * C_{n-1}^{λ+1}(x) for n >= 1
    // df/dx = sum_{k=1}^{N-1} c_k * 2λ * C_{k-1}^{λ+1}(x)

    T grad_x = T(0);
    if (N > 1 && lambda_ != T(0)) {
        // Compute C_k^{λ+1}(x) for k = 0..N-2
        T lambda_p1 = lambda_ + T(1);
        std::vector<T> C_p1(N - 1);
        C_p1[0] = T(1);

        if (N > 2) {
            C_p1[1] = T(2) * lambda_p1 * x;
        }

        for (int64_t k = 1; k < N - 2; ++k) {
            T a_k = T(2) * (T(k) + lambda_p1) / T(k + 1);
            T b_k = (T(k) + T(2) * lambda_p1 - T(1)) / T(k + 1);
            C_p1[k + 1] = a_k * x * C_p1[k] - b_k * C_p1[k - 1];
        }

        // df/dx = sum_{k=1}^{N-1} c_k * 2λ * C_{k-1}^{λ+1}(x)
        for (int64_t k = 1; k < N; ++k) {
            grad_x += coeffs[k] * T(2) * lambda_ * C_p1[k - 1];
        }
        grad_x *= grad_output;
    }

    // Compute grad_lambda
    // df/dλ = sum_k c_k * dC_k^{λ}(x)/dλ
    // We compute dC_k/dλ via forward recurrence alongside C_k
    std::vector<T> dC_dlambda(N);
    dC_dlambda[0] = T(0);  // dC_0/dλ = 0

    if (N > 1) {
        dC_dlambda[1] = T(2) * x;  // dC_1/dλ = d(2λx)/dλ = 2x
    }

    // Recurrence for dC_{k+1}/dλ:
    // C_{k+1} = A_k * x * C_k - B_k * C_{k-1}
    // where A_k = 2(k+λ)/(k+1), B_k = (k+2λ-1)/(k+1)
    // dA_k/dλ = 2/(k+1)
    // dB_k/dλ = 2/(k+1)
    // dC_{k+1}/dλ = dA_k/dλ * x * C_k + A_k * x * dC_k/dλ
    //             - dB_k/dλ * C_{k-1} - B_k * dC_{k-1}/dλ
    for (int64_t k = 1; k < N - 1; ++k) {
        T a_k = T(2) * (T(k) + lambda_) / T(k + 1);
        T b_k = (T(k) + T(2) * lambda_ - T(1)) / T(k + 1);
        T da_k_dlambda = T(2) / T(k + 1);
        T db_k_dlambda = T(2) / T(k + 1);

        dC_dlambda[k + 1] = da_k_dlambda * x * C[k] + a_k * x * dC_dlambda[k]
                         - db_k_dlambda * C[k - 1] - b_k * dC_dlambda[k - 1];
    }

    T grad_lambda = T(0);
    for (int64_t k = 0; k < N; ++k) {
        grad_lambda += coeffs[k] * dC_dlambda[k];
    }
    grad_lambda *= grad_output;

    return {grad_x, grad_lambda};
}

}  // namespace torchscience::kernel::polynomial
