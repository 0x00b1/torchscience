#pragma once

#include <cmath>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>

namespace torchscience::cpu::test_functions {

namespace {

inline void check_rosenbrock_input(const at::Tensor& x, const char* fn_name) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()),
        fn_name, " requires floating-point or complex input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1 && x.size(-1) >= 2,
        fn_name, " requires at least 2 dimensions in the last axis, got shape ",
        x.sizes()
    );
}

}  // anonymous namespace

namespace {

// ============================================================================
// Vectorized kernel implementations for real floating-point types
// ============================================================================

/**
 * Vectorized forward kernel for a single batch element.
 * Uses SIMD to process multiple consecutive pairs in parallel.
 */
template <typename scalar_t>
inline scalar_t rosenbrock_forward_vec_kernel(
    const scalar_t* x_ptr,
    scalar_t a_val,
    scalar_t b_val,
    int64_t n
) {
    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int64_t vec_size = Vec::size();
    const int64_t num_pairs = n - 1;

    Vec a_vec(a_val);
    Vec b_vec(b_val);
    Vec sum_vec(scalar_t(0));

    // Vectorized loop: process vec_size pairs per iteration
    // At iteration i, we compute pairs (x[i], x[i+1]), ..., (x[i+vec_size-1], x[i+vec_size])
    int64_t i = 0;
    for (; i <= num_pairs - vec_size; i += vec_size) {
        Vec x_i = Vec::loadu(x_ptr + i);
        Vec x_i_plus_1 = Vec::loadu(x_ptr + i + 1);

        Vec x_i_sq = x_i * x_i;
        Vec term1 = (a_vec - x_i) * (a_vec - x_i);
        Vec diff = x_i_plus_1 - x_i_sq;
        Vec term2 = b_vec * diff * diff;

        sum_vec = sum_vec + term1 + term2;
    }

    // Horizontal reduction
    scalar_t sum = at::vec::vec_reduce_all<scalar_t>(
        [](Vec& x, Vec& y) { return x + y; },
        sum_vec
    );

    // Scalar remainder loop
    for (; i < num_pairs; ++i) {
        scalar_t x_i = x_ptr[i];
        scalar_t x_i_plus_1 = x_ptr[i + 1];
        scalar_t x_i_sq = x_i * x_i;
        scalar_t term1 = (a_val - x_i) * (a_val - x_i);
        scalar_t diff = x_i_plus_1 - x_i_sq;
        scalar_t term2 = b_val * diff * diff;
        sum += term1 + term2;
    }

    return sum;
}

/**
 * Vectorized gradient kernel for a single batch element.
 * Computes df/dx_i for all i using SIMD operations.
 *
 * Gradient formula:
 *   df/dx_i = -2(a - x_i) - 4b*x_i*(x_{i+1} - x_i^2)  for contribution from term i
 *           + 2b*(x_i - x_{i-1}^2)                     for contribution from term i-1
 */
template <typename scalar_t>
inline void rosenbrock_gradient_vec_kernel(
    const scalar_t* x_ptr,
    scalar_t* grad_ptr,
    scalar_t a_val,
    scalar_t b_val,
    int64_t n
) {
    using Vec = at::vec::Vectorized<scalar_t>;
    constexpr int64_t vec_size = Vec::size();

    Vec a_vec(a_val);
    Vec b_vec(b_val);
    Vec neg_two(-scalar_t(2));
    Vec neg_four(-scalar_t(4));
    Vec two(scalar_t(2));

    // Initialize gradient to zero
    for (int64_t i = 0; i < n; ++i) {
        grad_ptr[i] = scalar_t(0);
    }

    // First pass: compute contribution from term i to grad[i]
    // df/dx_i (from term i) = -2(a - x_i) - 4b*x_i*(x_{i+1} - x_i^2)
    int64_t i = 0;
    for (; i <= (n - 1) - vec_size; i += vec_size) {
        Vec x_i = Vec::loadu(x_ptr + i);
        Vec x_i_plus_1 = Vec::loadu(x_ptr + i + 1);

        Vec x_i_sq = x_i * x_i;
        Vec diff = x_i_plus_1 - x_i_sq;

        Vec contrib = neg_two * (a_vec - x_i) + neg_four * b_vec * x_i * diff;

        Vec current = Vec::loadu(grad_ptr + i);
        (current + contrib).store(grad_ptr + i);
    }

    // Scalar remainder for first pass
    for (; i < n - 1; ++i) {
        scalar_t x_i = x_ptr[i];
        scalar_t x_i_plus_1 = x_ptr[i + 1];
        scalar_t x_i_sq = x_i * x_i;
        scalar_t diff = x_i_plus_1 - x_i_sq;
        grad_ptr[i] += scalar_t(-2) * (a_val - x_i) + scalar_t(-4) * b_val * x_i * diff;
    }

    // Second pass: compute contribution from term i-1 to grad[i]
    // df/dx_i (from term i-1) = 2b*(x_i - x_{i-1}^2)
    i = 1;
    for (; i <= n - vec_size; i += vec_size) {
        Vec x_i = Vec::loadu(x_ptr + i);
        Vec x_i_minus_1 = Vec::loadu(x_ptr + i - 1);

        Vec x_i_minus_1_sq = x_i_minus_1 * x_i_minus_1;
        Vec contrib = two * b_vec * (x_i - x_i_minus_1_sq);

        Vec current = Vec::loadu(grad_ptr + i);
        (current + contrib).store(grad_ptr + i);
    }

    // Scalar remainder for second pass
    for (; i < n; ++i) {
        scalar_t x_i = x_ptr[i];
        scalar_t x_i_minus_1 = x_ptr[i - 1];
        scalar_t x_i_minus_1_sq = x_i_minus_1 * x_i_minus_1;
        grad_ptr[i] += scalar_t(2) * b_val * (x_i - x_i_minus_1_sq);
    }
}

/**
 * Scalar gradient kernel for complex types.
 */
template <typename scalar_t>
inline void rosenbrock_gradient_scalar_kernel(
    const scalar_t* x_ptr,
    scalar_t* grad_ptr,
    scalar_t a_val,
    scalar_t b_val,
    int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        scalar_t grad_i = scalar_t(0);

        if (i < n - 1) {
            scalar_t x_i = x_ptr[i];
            scalar_t x_i_plus_1 = x_ptr[i + 1];
            scalar_t x_i_sq = x_i * x_i;
            scalar_t diff = x_i_plus_1 - x_i_sq;
            grad_i += scalar_t(-2) * (a_val - x_i);
            grad_i += scalar_t(-4) * b_val * x_i * diff;
        }

        if (i > 0) {
            scalar_t x_i_minus_1 = x_ptr[i - 1];
            scalar_t x_i = x_ptr[i];
            scalar_t x_i_minus_1_sq = x_i_minus_1 * x_i_minus_1;
            grad_i += scalar_t(2) * b_val * (x_i - x_i_minus_1_sq);
        }

        grad_ptr[i] = grad_i;
    }
}

/**
 * Scalar forward kernel for complex types.
 */
template <typename scalar_t>
inline scalar_t rosenbrock_forward_scalar_kernel(
    const scalar_t* x_ptr,
    scalar_t a_val,
    scalar_t b_val,
    int64_t n
) {
    scalar_t sum = scalar_t(0);
    for (int64_t i = 0; i < n - 1; ++i) {
        scalar_t x_i = x_ptr[i];
        scalar_t x_i_plus_1 = x_ptr[i + 1];
        scalar_t x_i_sq = x_i * x_i;
        scalar_t term1 = (a_val - x_i) * (a_val - x_i);
        scalar_t diff = x_i_plus_1 - x_i_sq;
        scalar_t term2 = b_val * diff * diff;
        sum += term1 + term2;
    }
    return sum;
}

// Compute analytical gradient of Rosenbrock function (internal helper)
inline at::Tensor compute_gradient(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});

    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (!a_is_scalar || !b_is_scalar) {
        // Fall back to ATen operations for tensor parameters
        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
        at::Tensor x_i_sq = at::pow(x_i, 2);
        at::Tensor diff = x_i_plus_1 - x_i_sq;

        at::Tensor grad = at::zeros_like(x);
        at::Tensor term1 = -2 * (a - x_i) - 4 * b * x_i * diff;
        grad.narrow(-1, 0, n - 1).add_(term1);

        at::Tensor x_prev_sq = at::pow(x.narrow(-1, 0, n - 1), 2);
        at::Tensor term2 = 2 * b * (x.narrow(-1, 1, n - 1) - x_prev_sq);
        grad.narrow(-1, 1, n - 1).add_(term2);

        return grad;
    }

    at::Tensor output = at::zeros({batch_size, n}, x.options());

    // Use vectorized kernels for real floating-point types
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rosenbrock_gradient_cpu_vec",
        [&]() {
            const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
            scalar_t* output_data = output.data_ptr<scalar_t>();
            scalar_t a_val = a.item<scalar_t>();
            scalar_t b_val = b.item<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                    const scalar_t* x_ptr = x_data + batch_idx * n;
                    scalar_t* grad_ptr = output_data + batch_idx * n;
                    rosenbrock_gradient_vec_kernel(x_ptr, grad_ptr, a_val, b_val, n);
                }
            });
        }
    );

    // Handle complex types separately with scalar kernel
    if (at::isComplexType(x.scalar_type())) {
        AT_DISPATCH_COMPLEX_TYPES(
            x.scalar_type(),
            "rosenbrock_gradient_cpu_complex",
            [&]() {
                const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
                scalar_t* output_data = output.data_ptr<scalar_t>();
                scalar_t a_val = a.item<scalar_t>();
                scalar_t b_val = b.item<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                        const scalar_t* x_ptr = x_data + batch_idx * n;
                        scalar_t* grad_ptr = output_data + batch_idx * n;
                        rosenbrock_gradient_scalar_kernel(x_ptr, grad_ptr, a_val, b_val, n);
                    }
                });
            }
        );
    }

    return output.view_as(x);
}

// Compute analytical Hessian of Rosenbrock function (internal helper)
// The Hessian is tridiagonal, so we compute only the non-zero elements
inline at::Tensor compute_hessian(
    const at::Tensor& x,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end() - 1);
    output_shape.push_back(n);
    output_shape.push_back(n);

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});

    bool b_is_scalar = (b.numel() == 1);

    if (!b_is_scalar) {
        at::Tensor H = at::zeros(output_shape, x.options());
        at::Tensor x_slice = x.narrow(-1, 0, n - 1);

        at::Tensor x_0 = x.select(-1, 0);
        at::Tensor x_1 = x.select(-1, 1);
        at::Tensor diag_0 = 2 + 12 * b * at::pow(x_0, 2) - 4 * b * x_1;

        at::Tensor x_mid = x.narrow(-1, 1, n - 2);
        at::Tensor x_mid_next = x.narrow(-1, 2, n - 2);
        at::Tensor diag_mid = 2 + 2 * b + 12 * b * at::pow(x_mid, 2) - 4 * b * x_mid_next;

        at::Tensor diag_last = 2 * b * at::ones_like(x.select(-1, -1));

        H.index_put_({at::indexing::Ellipsis, 0, 0}, diag_0);
        for (int64_t i = 1; i < n - 1; ++i) {
            H.index_put_({at::indexing::Ellipsis, i, i}, diag_mid.select(-1, i - 1));
        }
        H.index_put_({at::indexing::Ellipsis, n - 1, n - 1}, diag_last);

        at::Tensor off_diag = -4 * b * x_slice;
        for (int64_t i = 0; i < n - 1; ++i) {
            at::Tensor val = off_diag.select(-1, i);
            H.index_put_({at::indexing::Ellipsis, i, i + 1}, val);
            H.index_put_({at::indexing::Ellipsis, i + 1, i}, val);
        }

        return H;
    }

    at::Tensor output = at::zeros({batch_size, n, n}, x.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rosenbrock_hessian_cpu",
        [&]() {
            using Vec = at::vec::Vectorized<scalar_t>;
            constexpr int64_t vec_size = Vec::size();

            const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
            scalar_t* output_data = output.data_ptr<scalar_t>();
            scalar_t b_val = b.item<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                    const scalar_t* x_ptr = x_data + batch_idx * n;
                    scalar_t* H_ptr = output_data + batch_idx * n * n;

                    // Compute diagonal elements
                    // H[0,0] = 2 + 12*b*x_0^2 - 4*b*x_1
                    scalar_t x_0 = x_ptr[0];
                    scalar_t x_1 = x_ptr[1];
                    H_ptr[0] = scalar_t(2) + scalar_t(12) * b_val * x_0 * x_0 - scalar_t(4) * b_val * x_1;

                    // H[i,i] = 2 + 2*b + 12*b*x_i^2 - 4*b*x_{i+1} for i in [1, n-2]
                    for (int64_t i = 1; i < n - 1; ++i) {
                        scalar_t x_i = x_ptr[i];
                        scalar_t x_i_plus_1 = x_ptr[i + 1];
                        H_ptr[i * n + i] = scalar_t(2) + scalar_t(2) * b_val
                                           + scalar_t(12) * b_val * x_i * x_i
                                           - scalar_t(4) * b_val * x_i_plus_1;
                    }

                    // H[n-1, n-1] = 2*b
                    H_ptr[(n - 1) * n + (n - 1)] = scalar_t(2) * b_val;

                    // Off-diagonal elements: H[i, i+1] = H[i+1, i] = -4*b*x_i
                    for (int64_t i = 0; i < n - 1; ++i) {
                        scalar_t off_diag_val = scalar_t(-4) * b_val * x_ptr[i];
                        H_ptr[i * n + (i + 1)] = off_diag_val;
                        H_ptr[(i + 1) * n + i] = off_diag_val;
                    }
                }
            });
        }
    );

    return output.view(output_shape);
}

}  // anonymous namespace

/**
 * SIMD-vectorized CPU implementation of the Rosenbrock function.
 *
 * Uses at::vec::Vectorized<T> for SIMD acceleration on real floating-point types.
 * Falls back to scalar computation for complex types.
 */
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    check_rosenbrock_input(x, "rosenbrock");

    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end() - 1);
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});

    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    // Fall back to ATen for tensor parameters (broadcasting required)
    if (!a_is_scalar || !b_is_scalar) {
        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
        at::Tensor term1 = at::pow(a - x_i, 2);
        at::Tensor term2 = b * at::pow(x_i_plus_1 - at::pow(x_i, 2), 2);
        return at::sum(term1 + term2, -1);
    }

    at::Tensor output = at::empty({batch_size}, x.options());

    // Vectorized path for real floating-point types
    if (!at::isComplexType(x.scalar_type())) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16,
            at::kHalf,
            x.scalar_type(),
            "rosenbrock_cpu_vec",
            [&]() {
                const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
                scalar_t* output_data = output.data_ptr<scalar_t>();
                scalar_t a_val = a.item<scalar_t>();
                scalar_t b_val = b.item<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                        const scalar_t* x_ptr = x_data + batch_idx * n;
                        output_data[batch_idx] = rosenbrock_forward_vec_kernel(
                            x_ptr, a_val, b_val, n
                        );
                    }
                });
            }
        );
    } else {
        // Scalar path for complex types
        AT_DISPATCH_COMPLEX_TYPES(
            x.scalar_type(),
            "rosenbrock_cpu_complex",
            [&]() {
                const scalar_t* x_data = x_flat.data_ptr<scalar_t>();
                scalar_t* output_data = output.data_ptr<scalar_t>();
                scalar_t a_val = a.item<scalar_t>();
                scalar_t b_val = b.item<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
                        const scalar_t* x_ptr = x_data + batch_idx * n;
                        output_data[batch_idx] = rosenbrock_forward_scalar_kernel(
                            x_ptr, a_val, b_val, n
                        );
                    }
                });
            }
        );
    }

    if (x.dim() == 1) {
        return output.squeeze(0);
    }
    return output.view(output_shape);
}

/**
 * Backward pass for rosenbrock.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);

    // Compute gradient w.r.t. x using vectorized kernel
    at::Tensor grad_x_local = compute_gradient(x, a, b);
    at::Tensor grad_x = grad_output.unsqueeze(-1) * grad_x_local;

    // Compute gradient w.r.t. a: df/da = sum_i [2*(a - x_i)]
    at::Tensor x_i = x.narrow(-1, 0, n - 1);
    at::Tensor df_da = at::sum(2 * (a - x_i), -1);
    at::Tensor grad_a = grad_output * df_da;

    while (grad_a.dim() > a.dim()) {
        grad_a = grad_a.sum(0);
    }
    for (int64_t i = 0; i < a.dim(); ++i) {
        if (a.size(i) == 1 && grad_a.size(i) > 1) {
            grad_a = grad_a.sum(i, true);
        }
    }

    // Compute gradient w.r.t. b: df/db = sum_i [(x_{i+1} - x_i^2)^2]
    at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
    at::Tensor diff = x_i_plus_1 - at::pow(x_i, 2);
    at::Tensor df_db = at::sum(at::pow(diff, 2), -1);
    at::Tensor grad_b = grad_output * df_db;

    while (grad_b.dim() > b.dim()) {
        grad_b = grad_b.sum(0);
    }
    for (int64_t i = 0; i < b.dim(); ++i) {
        if (b.size(i) == 1 && grad_b.size(i) > 1) {
            grad_b = grad_b.sum(i, true);
        }
    }

    return std::make_tuple(grad_x, grad_a, grad_b);
}

/**
 * Double backward pass for rosenbrock.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward_backward(
    const at::Tensor& grad_grad_x,
    const at::Tensor& grad_grad_a,
    const at::Tensor& grad_grad_b,
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);

    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor grad_x = at::zeros_like(x);
    at::Tensor grad_a = at::zeros_like(a);
    at::Tensor grad_b = at::zeros_like(b);

    if (grad_grad_x.defined()) {
        at::Tensor H = compute_hessian(x, b);
        at::Tensor Hv = at::matmul(H, grad_grad_x.unsqueeze(-1)).squeeze(-1);
        grad_x = grad_x + grad_output.unsqueeze(-1) * Hv;

        at::Tensor grad_x_local = compute_gradient(x, a, b);
        grad_grad_output = grad_grad_output + at::sum(grad_grad_x * grad_x_local, -1);
    }

    if (grad_grad_a.defined()) {
        at::Tensor d2f_dxda = at::full({n - 1}, -2.0, x.options());
        at::Tensor contrib = grad_output * grad_grad_a;
        grad_x.narrow(-1, 0, n - 1).add_(contrib.unsqueeze(-1) * d2f_dxda);

        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor df_da = at::sum(2 * (a - x_i), -1);
        grad_grad_output = grad_grad_output + df_da * grad_grad_a;
    }

    if (grad_grad_b.defined()) {
        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
        at::Tensor diff = x_i_plus_1 - at::pow(x_i, 2);

        at::Tensor d2f_dxdb_i = -4 * x_i * diff;
        at::Tensor d2f_dxdb_i_plus_1 = 2 * diff;

        at::Tensor contrib = grad_output * grad_grad_b;
        grad_x.narrow(-1, 0, n - 1).add_(contrib.unsqueeze(-1) * d2f_dxdb_i);
        grad_x.narrow(-1, 1, n - 1).add_(contrib.unsqueeze(-1) * d2f_dxdb_i_plus_1);

        at::Tensor df_db = at::sum(at::pow(diff, 2), -1);
        grad_grad_output = grad_grad_output + df_db * grad_grad_b;
    }

    return std::make_tuple(grad_grad_output, grad_x, grad_a, grad_b);
}

}  // namespace torchscience::cpu::test_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("rosenbrock", &torchscience::cpu::test_functions::rosenbrock);
    module.impl("rosenbrock_backward", &torchscience::cpu::test_functions::rosenbrock_backward);
    module.impl("rosenbrock_backward_backward", &torchscience::cpu::test_functions::rosenbrock_backward_backward);
}
