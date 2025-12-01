// src/torchscience/csrc/cpu/statistics/hypothesis_test/anderson_darling.h
#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/anderson_darling.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Anderson-Darling test for normality - CPU implementation.
 *
 * Tests if a sample comes from a normal distribution.
 * Batches over all dimensions except the last (last dim = samples).
 *
 * @param input Input tensor where the last dimension contains samples
 * @return Tuple of (A^2 statistic, critical_values, significance_levels) tensors
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> anderson_darling(
    const at::Tensor& input
) {
    TORCH_CHECK(
        input.dim() >= 1,
        "anderson_darling: input must have at least 1 dimension, got ",
        input.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "anderson_darling: input must be floating-point, got ",
        input.scalar_type()
    );
    TORCH_CHECK(
        !input.requires_grad(),
        "anderson_darling: does not support gradients. Order statistics are not differentiable."
    );

    // Make input contiguous
    at::Tensor input_contig = input.contiguous();

    // Compute batch dimensions (all dims except last)
    int64_t n_samples = input_contig.size(-1);
    int64_t batch_size = input_contig.numel() / n_samples;

    // Output shape is input shape without the last dimension
    auto output_shape = input_contig.sizes().vec();
    output_shape.pop_back();

    // Create output tensors
    auto options = input_contig.options();
    at::Tensor statistic = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    // Critical values tensor: same batch shape + 5 critical values
    auto cv_shape = output_shape;
    cv_shape.push_back(5);
    at::Tensor critical_values = at::empty(cv_shape, options);

    // Significance levels are fixed: [0.15, 0.10, 0.05, 0.025, 0.01]
    at::Tensor significance_levels = at::tensor(
        {0.15, 0.10, 0.05, 0.025, 0.01},
        options
    );

    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "anderson_darling_cpu",
        [&] {
            const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
            scalar_t* stat_ptr = statistic.data_ptr<scalar_t>();
            scalar_t* cv_ptr = critical_values.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    auto [A2, cv] = kernel::anderson_darling<scalar_t>(
                        data_ptr + b * n_samples,
                        n_samples
                    );
                    stat_ptr[b] = A2;
                    for (int i = 0; i < 5; ++i) {
                        cv_ptr[b * 5 + i] = cv[i];
                    }
                }
            });
        }
    );

    return std::make_tuple(statistic, critical_values, significance_levels);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl(
        "anderson_darling",
        &torchscience::cpu::statistics::hypothesis_test::anderson_darling
    );
}
