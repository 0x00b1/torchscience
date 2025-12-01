// src/torchscience/csrc/cpu/statistics/hypothesis_test/shapiro_wilk.h
#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/shapiro_wilk.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Shapiro-Wilk test for normality - CPU implementation.
 *
 * Tests if a sample comes from a normal distribution.
 * Batches over all dimensions except the last (last dim = samples).
 *
 * @param input Input tensor where the last dimension contains samples
 * @return Tuple of (W-statistic, p-value) tensors
 */
inline std::tuple<at::Tensor, at::Tensor> shapiro_wilk(
    const at::Tensor& input
) {
    TORCH_CHECK(
        input.dim() >= 1,
        "shapiro_wilk: input must have at least 1 dimension, got ",
        input.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "shapiro_wilk: input must be floating-point, got ",
        input.scalar_type()
    );
    TORCH_CHECK(
        !input.requires_grad(),
        "shapiro_wilk: does not support gradients. Order statistics are not differentiable."
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
    at::Tensor pvalue = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    AT_DISPATCH_FLOATING_TYPES(
        input_contig.scalar_type(),
        "shapiro_wilk_cpu",
        [&] {
            const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
            scalar_t* stat_ptr = statistic.data_ptr<scalar_t>();
            scalar_t* pval_ptr = pvalue.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    auto [W, p] = kernel::shapiro_wilk<scalar_t>(
                        data_ptr + b * n_samples,
                        n_samples
                    );
                    stat_ptr[b] = W;
                    pval_ptr[b] = p;
                }
            });
        }
    );

    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl(
        "shapiro_wilk",
        &torchscience::cpu::statistics::hypothesis_test::shapiro_wilk
    );
}
