#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace torchscience::kernel::polynomial {

/**
 * Second-order backward pass for Probabilists' Hermite polynomial evaluation.
 *
 * Given:
 *   grad_coeffs[k] = grad_output * He_k(x)
 *   grad_x = grad_output * df/dx where df/dx = sum_{j=0}^{N-2} (j+1) * c_{j+1} * He_j(x)
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
std::tuple<T, T> hermite_polynomial_he_evaluate_backward_backward(
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

    // Compute all Hermite He polynomial values He_0(x), ..., He_{N-1}(x)
    std::vector<T> He(N);
    He[0] = T(1);
    if (N > 1) {
        He[1] = x;
    }
    for (int64_t k = 1; k < N - 1; ++k) {
        He[k + 1] = x * He[k] - T(k) * He[k - 1];
    }

    // Compute df/dx = sum_{j=0}^{N-2} (j+1) * c_{j+1} * He_j(x)
    T df_dx = T(0);
    for (int64_t j = 0; j < N - 1; ++j) {
        df_dx += T(j + 1) * coeffs[j + 1] * He[j];
    }

    // grad_grad_output contributions:
    // From grad_coeffs[k] = grad_output * He_k(x):
    //   ggo += sum_k gg_coeffs[k] * He_k(x)
    for (int64_t k = 0; k < N; ++k) {
        ggo += gg_coeffs[k] * He[k];
    }

    // From grad_x = grad_output * df_dx:
    //   ggo += gg_x * df_dx
    ggo += gg_x * df_dx;

    // g_coeffs contributions:
    // From grad_x = grad_output * sum_{j=0}^{N-2} (j+1) * c_{j+1} * He_j(x):
    //   d(grad_x)/d(c_m) = grad_output * m * He_{m-1}(x) for m >= 1
    //   g_coeffs[m] += gg_x * grad_output * m * He_{m-1}(x)
    for (int64_t m = 1; m < N; ++m) {
        g_coeffs[m] += gg_x * grad_output * T(m) * He[m - 1];
    }

    // g_x contributions:
    // From grad_coeffs[k] = grad_output * He_k(x):
    //   d(grad_coeffs[k])/d(x) = grad_output * He_k'(x) = grad_output * k * He_{k-1}(x) for k >= 1
    //   g_x += sum_{k=1}^{N-1} gg_coeffs[k] * grad_output * k * He_{k-1}(x)
    for (int64_t k = 1; k < N; ++k) {
        g_x += gg_coeffs[k] * grad_output * T(k) * He[k - 1];
    }

    // From grad_x = grad_output * df_dx:
    //   d(grad_x)/d(x) = grad_output * d(df_dx)/d(x)
    //   d(df_dx)/d(x) = sum_{j=0}^{N-2} (j+1) * c_{j+1} * He_j'(x)
    //                 = sum_{j=1}^{N-2} (j+1) * c_{j+1} * j * He_{j-1}(x)
    //                 = sum_{i=0}^{N-3} (i+1)(i+2) * c_{i+2} * He_i(x)
    T d2f_dx2 = T(0);
    for (int64_t i = 0; i < N - 2; ++i) {
        d2f_dx2 += T((i + 1) * (i + 2)) * coeffs[i + 2] * He[i];
    }
    g_x += gg_x * grad_output * d2f_dx2;

    return {ggo, g_x};
}

}  // namespace torchscience::kernel::polynomial
