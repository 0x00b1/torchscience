#pragma once

#include <cmath>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::distance {

namespace {

/**
 * Compute weighted Minkowski distance between two vectors.
 *
 * d_p(x, y; w) = ( sum_i w_i * |x_i - y_i|^p )^(1/p)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T minkowski_distance_pair(
    const T* x,
    const T* y,
    int64_t d,
    T p,
    const T* w
) {
    T sum = T(0);

    if (w == nullptr) {
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            T abs_diff = std::abs(diff);
            sum += std::pow(abs_diff, p);
        }
    } else {
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            T abs_diff = std::abs(diff);
            sum += w[i] * std::pow(abs_diff, p);
        }
    }

    if (p == T(1)) {
        return sum;
    } else if (p == T(2)) {
        return std::sqrt(sum);
    } else {
        return std::pow(sum, T(1) / p);
    }
}

/**
 * Compute gradients for weighted Minkowski distance.
 *
 * dd/dx_k = w_k * sign(x_k - y_k) * |x_k - y_k|^(p-1) / d^(p-1)
 * dd/dy_k = -dd/dx_k
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void minkowski_distance_backward_pair(
    T grad_out,
    const T* x,
    const T* y,
    int64_t d,
    T p,
    const T* w,
    T dist,
    T* grad_x,
    T* grad_y
) {
    if (dist == T(0)) {
        for (int64_t i = 0; i < d; ++i) {
            grad_x[i] = T(0);
            grad_y[i] = T(0);
        }
        return;
    }

    T dist_pow_pm1 = std::pow(dist, p - T(1));

    for (int64_t i = 0; i < d; ++i) {
        T diff = x[i] - y[i];

        if (diff == T(0)) {
            grad_x[i] = T(0);
            grad_y[i] = T(0);
            continue;
        }

        T abs_diff = std::abs(diff);
        T sign_diff = diff >= T(0) ? T(1) : T(-1);

        T abs_diff_pow_pm1;
        if (p == T(1)) {
            abs_diff_pow_pm1 = T(1);
        } else if (p == T(2)) {
            abs_diff_pow_pm1 = abs_diff;
        } else {
            abs_diff_pow_pm1 = std::pow(abs_diff, p - T(1));
        }

        T weight_i = (w != nullptr) ? w[i] : T(1);
        T grad_component = weight_i * sign_diff * abs_diff_pow_pm1 / dist_pow_pm1;

        grad_x[i] = grad_out * grad_component;
        grad_y[i] = -grad_out * grad_component;
    }
}

/**
 * Compute gradient for weight in weighted Minkowski distance.
 *
 * dd/dw_k = |x_k - y_k|^p / (p * d^(p-1))
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void minkowski_distance_weight_backward_pair(
    T grad_out,
    const T* x,
    const T* y,
    int64_t d,
    T p,
    T dist,
    T* grad_w
) {
    if (dist == T(0)) {
        for (int64_t i = 0; i < d; ++i) {
            grad_w[i] = T(0);
        }
        return;
    }

    T dist_pow_pm1 = std::pow(dist, p - T(1));
    T scale = T(1) / (p * dist_pow_pm1);

    for (int64_t i = 0; i < d; ++i) {
        T diff = x[i] - y[i];
        T abs_diff = std::abs(diff);

        T abs_diff_pow_p;
        if (p == T(1)) {
            abs_diff_pow_p = abs_diff;
        } else if (p == T(2)) {
            abs_diff_pow_p = abs_diff * abs_diff;
        } else {
            abs_diff_pow_p = std::pow(abs_diff, p);
        }

        grad_w[i] = grad_out * abs_diff_pow_p * scale;
    }
}

}  // anonymous namespace

inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    TORCH_CHECK(x.dim() == 2, "minkowski_distance: x must be 2D (m, d)");
    TORCH_CHECK(y.dim() == 2, "minkowski_distance: y must be 2D (n, d)");
    TORCH_CHECK(x.size(1) == y.size(1), "minkowski_distance: feature dimensions must match");
    TORCH_CHECK(p > 0, "minkowski_distance: p must be > 0");

    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t d = x.size(1);

    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();
    at::Tensor output = at::empty({m, n}, x.options());

    at::Tensor w_contig;
    bool has_weight = weight.has_value() && weight->defined();
    if (has_weight) {
        TORCH_CHECK(weight->dim() == 1, "minkowski_distance: weight must be 1D (d,)");
        TORCH_CHECK(weight->size(0) == d, "minkowski_distance: weight size must match feature dim");
        w_contig = weight->contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        x.scalar_type(),
        "minkowski_distance_cpu",
        [&]() {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* w_ptr = has_weight ? w_contig.data_ptr<scalar_t>() : nullptr;
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t p_val = static_cast<scalar_t>(p);

            at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    int64_t i = idx / n;
                    int64_t j = idx % n;
                    out_ptr[idx] = minkowski_distance_pair<scalar_t>(
                        x_ptr + i * d,
                        y_ptr + j * d,
                        d,
                        p_val,
                        w_ptr
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> minkowski_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& dist_output
) {
    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t d = x.size(1);

    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();
    at::Tensor grad_contig = grad_output.contiguous();
    at::Tensor dist_contig = dist_output.contiguous();

    at::Tensor grad_x = at::zeros_like(x);
    at::Tensor grad_y = at::zeros_like(y);
    at::Tensor grad_w;

    at::Tensor w_contig;
    bool has_weight = weight.has_value() && weight->defined();
    if (has_weight) {
        w_contig = weight->contiguous();
        grad_w = at::zeros_like(w_contig);
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        x.scalar_type(),
        "minkowski_distance_backward_cpu",
        [&]() {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();
            const scalar_t* dist_ptr = dist_contig.data_ptr<scalar_t>();
            const scalar_t* w_ptr = has_weight ? w_contig.data_ptr<scalar_t>() : nullptr;
            scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();
            scalar_t* grad_y_ptr = grad_y.data_ptr<scalar_t>();
            scalar_t* grad_w_ptr = has_weight ? grad_w.data_ptr<scalar_t>() : nullptr;
            scalar_t p_val = static_cast<scalar_t>(p);

            std::vector<scalar_t> temp_grad_x(d);
            std::vector<scalar_t> temp_grad_y(d);
            std::vector<scalar_t> temp_grad_w(d);

            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t grad_val = grad_ptr[i * n + j];
                    scalar_t dist_val = dist_ptr[i * n + j];

                    minkowski_distance_backward_pair<scalar_t>(
                        grad_val,
                        x_ptr + i * d,
                        y_ptr + j * d,
                        d,
                        p_val,
                        w_ptr,
                        dist_val,
                        temp_grad_x.data(),
                        temp_grad_y.data()
                    );

                    for (int64_t k = 0; k < d; ++k) {
                        grad_x_ptr[i * d + k] += temp_grad_x[k];
                        grad_y_ptr[j * d + k] += temp_grad_y[k];
                    }

                    if (has_weight) {
                        minkowski_distance_weight_backward_pair<scalar_t>(
                            grad_val,
                            x_ptr + i * d,
                            y_ptr + j * d,
                            d,
                            p_val,
                            dist_val,
                            temp_grad_w.data()
                        );

                        for (int64_t k = 0; k < d; ++k) {
                            grad_w_ptr[k] += temp_grad_w[k];
                        }
                    }
                }
            }
        }
    );

    return std::make_tuple(grad_x, grad_y, grad_w);
}

}  // namespace torchscience::cpu::distance

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("minkowski_distance", &torchscience::cpu::distance::minkowski_distance);
    module.impl("minkowski_distance_backward", &torchscience::cpu::distance::minkowski_distance_backward);
}
