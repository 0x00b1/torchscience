#pragma once

#include <cmath>
#include <vector>

#include "kullback_leibler_divergence.h"

namespace torchscience::kernel::information_theory {

/**
 * Compute second-order gradient of pointwise mutual information.
 *
 * The first derivative is (from backward):
 *   grad_joint[x,y] = scale * (grad_output[x,y]/p(x,y) - g_x[x]/p(x) - g_y[y]/p(y))
 *
 * where g_x[x] = sum_y grad_output[x,y] and g_y[y] = sum_x grad_output[x,y].
 *
 * For the backward_backward, we need gradients w.r.t. grad_output and joint:
 *
 * 1. d(grad_joint[x,y])/d(grad_output[x',y']):
 *    - When (x,y) = (x',y'): 1/p(x,y)
 *    - When x = x' (same row): -1/p(x)
 *    - When y = y' (same column): -1/p(y)
 *
 * 2. d(grad_joint[x',y'])/d(joint[x,y]):
 *    This involves differentiating through p(x,y), p(x), and p(y).
 *    The full Hessian is complex; we compute the Hessian-vector product.
 *
 * @param gg_joint Gradient w.r.t. grad_joint from upstream [size_x, size_y]
 * @param grad_output Original upstream gradient [size_x, size_y]
 * @param joint Joint distribution [size_x, size_y]
 * @param p_x Marginal p(x) [size_x]
 * @param p_y Marginal p(y) [size_y]
 * @param size_x Size of X dimension
 * @param size_y Size of Y dimension
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_grad_output Output: gradient w.r.t. grad_output [size_x, size_y]
 * @param grad_joint Output: gradient w.r.t. joint [size_x, size_y]
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void pointwise_mutual_information_backward_backward_kernel(
    const T* gg_joint,
    const T* grad_output,
    const T* joint,
    const T* p_x,
    const T* p_y,
    int64_t size_x,
    int64_t size_y,
    T log_base_scale,
    T* grad_grad_output,
    T* grad_joint
) {
    T eps = get_eps<T>();

    // Compute sums of gg_joint along rows and columns
    // gg_row_sum[i] = sum_j gg_joint[i,j]
    // gg_col_sum[j] = sum_i gg_joint[i,j]
    std::vector<T> gg_row_sum(size_x, T(0));
    std::vector<T> gg_col_sum(size_y, T(0));

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            T gg = gg_joint ? gg_joint[i * size_y + j] : T(0);
            gg_row_sum[i] += gg;
            gg_col_sum[j] += gg;
        }
    }

    // Compute sums of grad_output along rows and columns
    // g_x[i] = sum_j grad_output[i,j]
    // g_y[j] = sum_i grad_output[i,j]
    std::vector<T> g_x(size_x, T(0));
    std::vector<T> g_y(size_y, T(0));

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            T g_out = grad_output[i * size_y + j];
            g_x[i] += g_out;
            g_y[j] += g_out;
        }
    }

    // Compute auxiliary terms for grad_joint computation
    // gg_g_over_px2[i] = sum_j (gg_joint[i,j] * g_x[i]) / p_x[i]^2
    //                  = (g_x[i] / p_x[i]^2) * sum_j gg_joint[i,j]
    //                  = (g_x[i] / p_x[i]^2) * gg_row_sum[i]
    // gg_g_over_py2[j] = sum_i (gg_joint[i,j] * g_y[j]) / p_y[j]^2
    //                  = (g_y[j] / p_y[j]^2) * sum_i gg_joint[i,j]
    //                  = (g_y[j] / p_y[j]^2) * gg_col_sum[j]
    std::vector<T> gg_g_over_px2(size_x, T(0));
    std::vector<T> gg_g_over_py2(size_y, T(0));

    for (int64_t i = 0; i < size_x; ++i) {
        T px = p_x[i] > eps ? p_x[i] : eps;
        gg_g_over_px2[i] = (g_x[i] / (px * px)) * gg_row_sum[i];
    }

    for (int64_t j = 0; j < size_y; ++j) {
        T py = p_y[j] > eps ? p_y[j] : eps;
        gg_g_over_py2[j] = (g_y[j] / (py * py)) * gg_col_sum[j];
    }

    for (int64_t i = 0; i < size_x; ++i) {
        for (int64_t j = 0; j < size_y; ++j) {
            int64_t idx = i * size_y + j;
            T p_xy = joint[idx] > eps ? joint[idx] : eps;
            T px = p_x[i] > eps ? p_x[i] : eps;
            T py = p_y[j] > eps ? p_y[j] : eps;

            T gg = gg_joint ? gg_joint[idx] : T(0);
            T g_out = grad_output[idx];

            // Gradient w.r.t. grad_output[i,j]
            // grad_grad_output[i,j] = sum_{x',y'} gg_joint[x',y'] * d(grad_joint[x',y'])/d(grad_output[i,j])
            //
            // d(grad_joint[x',y'])/d(grad_output[i,j]):
            //   - When (x',y') = (i,j): 1/p(i,j)
            //   - When x' = i, y' != j: -1/p(i)  (via g_x contribution)
            //   - When x' != i, y' = j: -1/p(j)  (via g_y contribution)
            //   - Otherwise: 0
            //
            // So: grad_grad_output[i,j] = scale * (
            //     gg[i,j] / p(i,j)
            //     - sum_{y' != j} gg[i,y'] / p(i)
            //     - sum_{x' != i} gg[x',j] / p(j)
            // )
            //
            // = scale * (gg[i,j] / p(i,j) - (gg_row_sum[i] - gg[i,j]) / p(i) - (gg_col_sum[j] - gg[i,j]) / p(j))
            // = scale * (gg[i,j] / p(i,j) - gg_row_sum[i] / p(i) + gg[i,j] / p(i) - gg_col_sum[j] / p(j) + gg[i,j] / p(j))
            //
            // Actually the simpler form works:
            // = scale * (gg[i,j] / p(i,j) - gg_row_sum[i] / p(i) - gg_col_sum[j] / p(j))
            // since the same-index terms cancel appropriately in the derivation.
            if (grad_grad_output) {
                grad_grad_output[idx] = (gg / p_xy - gg_row_sum[i] / px - gg_col_sum[j] / py) * log_base_scale;
            }

            // Gradient w.r.t. joint[i,j]
            // We need d(grad_joint[x',y'])/d(joint[i,j]) contracted with gg_joint[x',y']
            //
            // grad_joint[x',y'] = scale * (g_out[x',y']/p(x',y') - g_x[x']/p(x') - g_y[y']/p(y'))
            //
            // Differentiating w.r.t. joint[i,j]:
            // d(p(x',y'))/d(joint[i,j]) = delta(x'=i, y'=j)
            // d(p(x'))/d(joint[i,j]) = delta(x'=i)  (p(x') = sum_y p(x',y))
            // d(p(y'))/d(joint[i,j]) = delta(y'=j)  (p(y') = sum_x p(x,y'))
            //
            // d(grad_joint[x',y'])/d(joint[i,j]) = scale * (
            //   - delta(x'=i, y'=j) * g_out[x',y'] / p(x',y')^2
            //   + delta(x'=i) * g_x[x'] / p(x')^2
            //   + delta(y'=j) * g_y[y'] / p(y')^2
            // )
            //
            // Contracting with gg_joint[x',y']:
            // grad_joint[i,j] = scale * (
            //   - gg[i,j] * g_out[i,j] / p(i,j)^2
            //   + sum_{y'} gg[i,y'] * g_x[i] / p(i)^2
            //   + sum_{x'} gg[x',j] * g_y[j] / p(j)^2
            // )
            //
            // = scale * (
            //   - gg[i,j] * g_out[i,j] / p(i,j)^2
            //   + gg_row_sum[i] * g_x[i] / p(i)^2
            //   + gg_col_sum[j] * g_y[j] / p(j)^2
            // )
            if (grad_joint) {
                grad_joint[idx] = log_base_scale * (
                    - gg * g_out / (p_xy * p_xy)
                    + gg_g_over_px2[i]
                    + gg_g_over_py2[j]
                );
            }
        }
    }
}

}  // namespace torchscience::kernel::information_theory
