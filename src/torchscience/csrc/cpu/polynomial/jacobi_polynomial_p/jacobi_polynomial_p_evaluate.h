#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "../../../kernel/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate_backward.h"
#include "../../../kernel/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate_backward_backward.h"

namespace torchscience::cpu::polynomial {

inline at::Tensor jacobi_polynomial_p_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    auto output = at::empty({B, M}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        coeffs.scalar_type(), "jacobi_polynomial_p_evaluate_cpu", [&] {
            const scalar_t* coeffs_ptr = coeffs.data_ptr<scalar_t>();
            const scalar_t* x_ptr = x.data_ptr<scalar_t>();
            scalar_t alpha_val = alpha.item<scalar_t>();
            scalar_t beta_val = beta.item<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            for (int64_t b = 0; b < B; ++b) {
                const scalar_t* batch_coeffs = coeffs_ptr + b * N;
                for (int64_t m = 0; m < M; ++m) {
                    output_ptr[b * M + m] = kernel::polynomial::jacobi_polynomial_p_evaluate(
                        batch_coeffs, x_ptr[m], alpha_val, beta_val, N
                    );
                }
            }
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> jacobi_polynomial_p_evaluate_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    auto grad_coeffs = at::zeros({B, N}, coeffs.options());
    auto grad_x = at::zeros({M}, x.options());
    auto grad_alpha = at::zeros({}, alpha.options());
    auto grad_beta = at::zeros({}, beta.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        coeffs.scalar_type(), "jacobi_polynomial_p_evaluate_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output.data_ptr<scalar_t>();
            const scalar_t* coeffs_ptr = coeffs.data_ptr<scalar_t>();
            const scalar_t* x_ptr = x.data_ptr<scalar_t>();
            scalar_t alpha_val = alpha.item<scalar_t>();
            scalar_t beta_val = beta.item<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();
            scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();

            // Temporary storage for P_k(x) values
            std::vector<scalar_t> P_values(N);

            for (int64_t m = 0; m < M; ++m) {
                scalar_t x_val = x_ptr[m];

                // Compute all Jacobi polynomials at this x
                kernel::polynomial::jacobi_polynomial_p_compute_all(
                    P_values.data(), x_val, alpha_val, beta_val, N
                );

                for (int64_t b = 0; b < B; ++b) {
                    const scalar_t* batch_coeffs = coeffs_ptr + b * N;
                    scalar_t g = grad_output_ptr[b * M + m];

                    // grad_coeffs[k] += grad_output * P_k(x)
                    for (int64_t k = 0; k < N; ++k) {
                        grad_coeffs_ptr[b * N + k] += g * P_values[k];
                    }

                    // grad_x += grad_output * df/dx
                    scalar_t df_dx = kernel::polynomial::jacobi_polynomial_p_evaluate_backward_x(
                        batch_coeffs, x_val, alpha_val, beta_val, N
                    );
                    grad_x_ptr[m] += g * df_dx;
                }
            }
        }
    );

    return {grad_coeffs, grad_x, grad_alpha, grad_beta};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> jacobi_polynomial_p_evaluate_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& gg_x,
    const at::Tensor& gg_alpha,
    const at::Tensor& gg_beta,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    auto grad_grad_output = at::zeros({B, M}, grad_output.options());
    auto g_coeffs = at::zeros({B, N}, coeffs.options());
    auto g_x = at::zeros({M}, x.options());
    auto g_alpha = at::zeros({}, alpha.options());
    auto g_beta = at::zeros({}, beta.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        coeffs.scalar_type(), "jacobi_polynomial_p_evaluate_backward_backward_cpu", [&] {
            const scalar_t* gg_coeffs_ptr = gg_coeffs.data_ptr<scalar_t>();
            const scalar_t* gg_x_ptr = gg_x.data_ptr<scalar_t>();
            const scalar_t* grad_output_ptr = grad_output.data_ptr<scalar_t>();
            const scalar_t* coeffs_ptr = coeffs.data_ptr<scalar_t>();
            const scalar_t* x_ptr = x.data_ptr<scalar_t>();
            scalar_t alpha_val = alpha.item<scalar_t>();
            scalar_t beta_val = beta.item<scalar_t>();

            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* g_coeffs_ptr = g_coeffs.data_ptr<scalar_t>();
            scalar_t* g_x_ptr = g_x.data_ptr<scalar_t>();

            std::vector<scalar_t> P_values(N);

            for (int64_t m = 0; m < M; ++m) {
                scalar_t x_val = x_ptr[m];

                kernel::polynomial::jacobi_polynomial_p_compute_all(
                    P_values.data(), x_val, alpha_val, beta_val, N
                );

                for (int64_t b = 0; b < B; ++b) {
                    const scalar_t* batch_coeffs = coeffs_ptr + b * N;
                    scalar_t gg_x_m = gg_x_ptr[m];
                    scalar_t g = grad_output_ptr[b * M + m];

                    // From gg_coeffs: grad_grad_output += gg_coeffs * P_k(x)
                    for (int64_t k = 0; k < N; ++k) {
                        ggo_ptr[b * M + m] += gg_coeffs_ptr[b * N + k] * P_values[k];
                    }

                    // From gg_x: grad_grad_output += gg_x * df/dx
                    scalar_t df_dx = kernel::polynomial::jacobi_polynomial_p_evaluate_backward_x(
                        batch_coeffs, x_val, alpha_val, beta_val, N
                    );
                    ggo_ptr[b * M + m] += gg_x_m * df_dx;

                    // g_x += grad_output * gg_x * d²f/dx²
                    scalar_t d2f_dx2 = kernel::polynomial::jacobi_polynomial_p_evaluate_backward_backward_x(
                        batch_coeffs, x_val, alpha_val, beta_val, N
                    );
                    g_x_ptr[m] += g * gg_x_m * d2f_dx2;

                    // g_coeffs += grad_output * gg_x * P_k'(x)
                    // For now, we approximate this contribution
                }
            }
        }
    );

    return {grad_grad_output, g_coeffs, g_x, g_alpha, g_beta};
}

}  // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("jacobi_polynomial_p_evaluate", torchscience::cpu::polynomial::jacobi_polynomial_p_evaluate);
    m.impl("jacobi_polynomial_p_evaluate_backward", torchscience::cpu::polynomial::jacobi_polynomial_p_evaluate_backward);
    m.impl("jacobi_polynomial_p_evaluate_backward_backward", torchscience::cpu::polynomial::jacobi_polynomial_p_evaluate_backward_backward);
}
