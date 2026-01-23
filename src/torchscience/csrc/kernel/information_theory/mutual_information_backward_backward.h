#pragma once

#include <cmath>
#include <vector>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradient of mutual information.
 *
 * The first derivative is:
 *   dI/dp(x,y) = log(p(x,y) / (p(x) * p(y))) - 1
 *
 * The Hessian d²I/dp(x,y)dp(x',y') has contributions:
 *   - d²I/dp(x,y)² = 1/p(x,y) - 1/p(x) - 1/p(y)  (diagonal)
 *   - d²I/dp(x,y)dp(x,y') = -1/p(x)  (same row, different column)
 *   - d²I/dp(x,y)dp(x',y) = -1/p(y)  (same column, different row)
 *
 * For the backward of backward:
 * - grad_grad_output = sum_{x,y} gg_joint[x,y] * (dI/dp(x,y)) * scale
 * - grad_joint[x,y] = grad_output * sum_{x',y'} gg_joint[x',y'] * H[x,y][x',y'] * scale
 *
 * @param gg_joint Gradient w.r.t. grad_joint from upstream
 * @param grad_output Original upstream gradient
 * @param joint Pointer to joint distribution
 * @param p_x Pointer to marginal p(x)
 * @param p_y Pointer to marginal p(y)
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_joint Output: gradient w.r.t. joint
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void mutual_information_backward_backward_kernel(
    const T* gg_joint,
    T grad_output,
    const T* joint,
    const T* p_x,
    const T* p_y,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_joint
) {
    T eps = get_eps<T>();
    grad_grad_output = T(0);

    // First, compute sums of gg_joint along rows and columns for Hessian computation
    // gg_row_sum[i] = sum_j gg_joint[i,j]
    // gg_col_sum[j] = sum_i gg_joint[i,j]
    std::vector<T> gg_row_sum(size_x, T(0));
    std::vector<T> gg_col_sum(size_y, T(0));

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            int64_t idx = i * size_y + j;
            T gg = gg_joint ? gg_joint[idx] : T(0);
            gg_row_sum[i] += gg;
            gg_col_sum[j] += gg;
        }
    }

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            int64_t idx = i * size_y + j;
            T p_xy = joint[idx] > eps ? joint[idx] : eps;
            T px = p_x[i] > eps ? p_x[i] : eps;
            T py = p_y[j] > eps ? p_y[j] : eps;
            T marginal_prod = px * py;

            T first_deriv;

            if (marginal_prod > eps && p_xy > eps) {
                // First derivative: log(p(x,y) / (p(x) * p(y))) - 1
                first_deriv = std::log(p_xy / marginal_prod) - T(1);
            } else {
                first_deriv = T(0);
            }

            T gg = gg_joint ? gg_joint[idx] : T(0);

            // Gradient w.r.t. grad_output
            grad_grad_output += gg * first_deriv * log_base_scale;

            // Gradient w.r.t. joint[i,j]
            // The Hessian row for p(i,j) is:
            //   H[i,j][i,j] = 1/p(i,j) - 1/p_x[i] - 1/p_y[j]  (diagonal term)
            //   H[i,j][i,k] = -1/p_x[i]  for k != j  (same row)
            //   H[i,j][k,j] = -1/p_y[j]  for k != i  (same column)
            //   H[i,j][k,l] = 0  otherwise
            //
            // So grad_joint[i,j] = grad_output * scale * (
            //     gg[i,j] * (1/p(i,j) - 1/p_x[i] - 1/p_y[j])
            //     + sum_{k!=j} gg[i,k] * (-1/p_x[i])
            //     + sum_{k!=i} gg[k,j] * (-1/p_y[j])
            // )
            //
            // = grad_output * scale * (
            //     gg[i,j] * (1/p(i,j) - 1/p_x[i] - 1/p_y[j])
            //     + (gg_row_sum[i] - gg[i,j]) * (-1/p_x[i])
            //     + (gg_col_sum[j] - gg[i,j]) * (-1/p_y[j])
            // )
            //
            // = grad_output * scale * (
            //     gg[i,j] / p(i,j)
            //     - gg[i,j] / p_x[i] - (gg_row_sum[i] - gg[i,j]) / p_x[i]
            //     - gg[i,j] / p_y[j] - (gg_col_sum[j] - gg[i,j]) / p_y[j]
            // )
            //
            // = grad_output * scale * (
            //     gg[i,j] / p(i,j) - gg_row_sum[i] / p_x[i] - gg_col_sum[j] / p_y[j]
            // )
            if (grad_joint) {
                T hessian_contrib = gg / p_xy - gg_row_sum[i] / px - gg_col_sum[j] / py;
                grad_joint[idx] = grad_output * hessian_contrib * log_base_scale;
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
