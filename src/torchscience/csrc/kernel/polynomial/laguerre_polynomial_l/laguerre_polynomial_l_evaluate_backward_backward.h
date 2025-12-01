#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace torchscience::kernel::polynomial {

/**
 * Second-order backward pass for Laguerre polynomial evaluation.
 *
 * Given:
 *   grad_coeffs[k] = grad_output * L_k(x)
 *   grad_x = grad_output * df/dx where df/dx = -sum_j L_j(x) * suffix_sum[j]
 *
 * Computes gradients of grad_coeffs and grad_x with respect to:
 *   - grad_output -> returns grad_grad_output
 *   - coeffs -> returns g_coeffs
 *   - x -> returns g_x
 *
 * @param g_coeffs Output: gradient w.r.t. coefficients (size N)
 * @param gg_coeffs Upstream gradient for grad_coeffs (size N)
 * @param gg_x Upstream gradient for grad_x (scalar)
 * @param grad_output Original upstream gradient
 * @param coeffs Polynomial coefficients (size N)
 * @param x Evaluation point
 * @param N Number of coefficients
 * @return Tuple of (grad_grad_output, g_x)
 */
template <typename T>
std::tuple<T, T> laguerre_polynomial_l_evaluate_backward_backward(
    T* g_coeffs,
    const T* gg_coeffs,
    T gg_x,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    T ggo = T(0);  // grad_grad_output
    T g_x = T(0);

    // Initialize g_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        g_coeffs[k] = T(0);
    }

    if (N == 0) {
        return {ggo, g_x};
    }

    // Compute all Laguerre polynomial values L_0(x), ..., L_{N-1}(x)
    std::vector<T> L(N);
    L[0] = T(1);
    if (N > 1) {
        L[1] = T(1) - x;
    }
    for (int64_t k = 1; k < N - 1; ++k) {
        L[k + 1] = ((T(2 * k + 1) - x) * L[k] - T(k) * L[k - 1]) / T(k + 1);
    }

    // Compute prefix sums of L: prefix_L[k] = sum_{j=0}^{k-1} L_j(x) = -L_k'(x)
    // L_k'(x) = -sum_{j=0}^{k-1} L_j(x), so prefix_L[k] = -L_k'(x)
    std::vector<T> prefix_L(N);
    prefix_L[0] = T(0);  // L_0'(x) = 0
    for (int64_t k = 1; k < N; ++k) {
        prefix_L[k] = prefix_L[k - 1] + L[k - 1];
    }

    // Compute suffix sums of coeffs: suffix_sum[j] = sum_{k=j+1}^{N-1} c_k
    std::vector<T> suffix_sum(N);
    suffix_sum[N - 1] = T(0);
    for (int64_t j = N - 2; j >= 0; --j) {
        suffix_sum[j] = suffix_sum[j + 1] + coeffs[j + 1];
    }

    // Compute df/dx = -sum_j L_j(x) * suffix_sum[j]
    T df_dx = T(0);
    for (int64_t j = 0; j < N; ++j) {
        df_dx -= L[j] * suffix_sum[j];
    }

    // grad_grad_output contributions:
    // From grad_coeffs[k] = grad_output * L_k(x):
    //   ggo += sum_k gg_coeffs[k] * L_k(x)
    for (int64_t k = 0; k < N; ++k) {
        ggo += gg_coeffs[k] * L[k];
    }

    // From grad_x = grad_output * df_dx:
    //   ggo += gg_x * df_dx
    ggo += gg_x * df_dx;

    // g_coeffs contributions:
    // From grad_x = grad_output * (-sum_j L_j(x) * suffix_sum[j]):
    //   d(grad_x)/d(c_m) = grad_output * (-sum_{j=0}^{m-1} L_j(x))
    //                    = grad_output * (-prefix_L[m])
    //   g_coeffs[m] += gg_x * grad_output * (-prefix_L[m])
    for (int64_t m = 0; m < N; ++m) {
        g_coeffs[m] += gg_x * grad_output * (-prefix_L[m]);
    }

    // g_x contributions:
    // From grad_coeffs[k] = grad_output * L_k(x):
    //   d(grad_coeffs[k])/d(x) = grad_output * L_k'(x) = grad_output * (-prefix_L[k])
    //   g_x += sum_k gg_coeffs[k] * grad_output * (-prefix_L[k])
    for (int64_t k = 0; k < N; ++k) {
        g_x += gg_coeffs[k] * grad_output * (-prefix_L[k]);
    }

    // From grad_x = grad_output * df_dx:
    //   d(grad_x)/d(x) = grad_output * d(df_dx)/d(x)
    //   d(df_dx)/d(x) = -sum_j L_j'(x) * suffix_sum[j]
    //                 = -sum_j (-prefix_L[j]) * suffix_sum[j]
    //                 = sum_j prefix_L[j] * suffix_sum[j]
    T d2f_dx2 = T(0);
    for (int64_t j = 0; j < N; ++j) {
        d2f_dx2 += prefix_L[j] * suffix_sum[j];
    }
    g_x += gg_x * grad_output * d2f_dx2;

    return {ggo, g_x};
}

}  // namespace torchscience::kernel::polynomial
