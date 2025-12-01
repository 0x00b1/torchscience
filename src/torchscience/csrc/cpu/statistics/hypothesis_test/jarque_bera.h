#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/jarque_bera.h"
#include "../../../kernel/statistics/hypothesis_test/jarque_bera_backward.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Jarque-Bera test for normality - CPU implementation.
 *
 * Tests if a sample comes from a normal distribution by checking
 * whether the skewness and kurtosis match that of a normal distribution.
 *
 * @param input Input tensor where the last dimension contains samples
 * @return Tuple of (JB-statistic, p-value) tensors
 */
inline std::tuple<at::Tensor, at::Tensor> jarque_bera(const at::Tensor& input) {
    TORCH_CHECK(
        input.dim() >= 1,
        "jarque_bera: input must have at least 1 dimension, got ",
        input.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "jarque_bera: input must be floating-point, got ",
        input.scalar_type()
    );

    at::Tensor input_contig = input.contiguous();
    int64_t n_samples = input_contig.size(-1);
    int64_t batch_size = input_contig.numel() / n_samples;

    auto output_shape = input_contig.sizes().vec();
    output_shape.pop_back();

    auto options = input_contig.options();
    at::Tensor statistic = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor pvalue = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "jarque_bera_cpu",
        [&] {
            const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
            scalar_t* stat_ptr = statistic.data_ptr<scalar_t>();
            scalar_t* pval_ptr = pvalue.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    auto [s, p] = kernel::jarque_bera<scalar_t>(
                        data_ptr + b * n_samples,
                        n_samples
                    );
                    stat_ptr[b] = s;
                    pval_ptr[b] = p;
                }
            });
        }
    );

    return std::make_tuple(statistic, pvalue);
}

/**
 * Backward pass for Jarque-Bera test - CPU implementation.
 *
 * @param grad_statistic Gradient with respect to JB statistic
 * @param input Original input tensor
 * @return Gradient with respect to input
 */
inline at::Tensor jarque_bera_backward(
    const at::Tensor& grad_statistic,
    const at::Tensor& input
) {
    TORCH_CHECK(
        input.dim() >= 1,
        "jarque_bera_backward: input must have at least 1 dimension"
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "jarque_bera_backward: input must be floating-point"
    );

    at::Tensor input_contig = input.contiguous();
    at::Tensor grad_contig = grad_statistic.contiguous();

    int64_t n_samples = input_contig.size(-1);
    int64_t batch_size = input_contig.numel() / n_samples;

    at::Tensor grad_input = at::empty_like(input_contig);

    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "jarque_bera_backward_cpu",
        [&] {
            const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
            const scalar_t* grad_stat_ptr = grad_contig.data_ptr<scalar_t>();
            scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    kernel::jarque_bera_backward<scalar_t>(
                        grad_stat_ptr[b],
                        data_ptr + b * n_samples,
                        grad_input_ptr + b * n_samples,
                        n_samples
                    );
                }
            });
        }
    );

    return grad_input;
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("jarque_bera", &torchscience::cpu::statistics::hypothesis_test::jarque_bera);
    m.impl("jarque_bera_backward", &torchscience::cpu::statistics::hypothesis_test::jarque_bera_backward);
}
