// src/torchscience/csrc/cpu/statistics/hypothesis_test/paired_t_test.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/paired_t_test.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Paired t-test CPU implementation.
 *
 * Tests if the mean difference between paired observations is zero.
 * Batches over all dimensions except the last (last dim = paired samples).
 * Shapes of input1 and input2 must match exactly.
 *
 * @param input1 First sample tensor (last dim = samples)
 * @param input2 Second sample tensor (must match input1 shape exactly)
 * @param alternative Alternative hypothesis: "two-sided", "less", or "greater"
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom) tensors
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> paired_t_test(
    const at::Tensor& input1,
    const at::Tensor& input2,
    c10::string_view alternative
) {
    TORCH_CHECK(
        input1.dim() >= 1,
        "paired_t_test: input1 must have at least 1 dimension, got ",
        input1.dim(), "D"
    );
    TORCH_CHECK(
        input2.dim() >= 1,
        "paired_t_test: input2 must have at least 1 dimension, got ",
        input2.dim(), "D"
    );
    TORCH_CHECK(
        input1.sizes() == input2.sizes(),
        "paired_t_test: input1 and input2 must have the same shape, got ",
        input1.sizes(), " and ", input2.sizes()
    );
    TORCH_CHECK(
        at::isFloatingType(input1.scalar_type()),
        "paired_t_test: input1 must be floating-point, got ",
        input1.scalar_type()
    );
    TORCH_CHECK(
        input1.scalar_type() == input2.scalar_type(),
        "paired_t_test: input1 and input2 must have the same dtype, got ",
        input1.scalar_type(), " and ", input2.scalar_type()
    );

    // Parse alternative hypothesis
    kernel::Alternative alt = kernel::parse_alternative(
        std::string(alternative).c_str()
    );

    // Make inputs contiguous
    at::Tensor input1_contig = input1.contiguous();
    at::Tensor input2_contig = input2.contiguous();

    // Compute dimensions
    int64_t n_samples = input1_contig.size(-1);
    int64_t batch_size = input1_contig.numel() / n_samples;

    // Output shape is input shape without the last dimension
    auto output_shape = input1_contig.sizes().vec();
    output_shape.pop_back();

    // Create output tensors
    auto options = input1_contig.options();
    at::Tensor t_stat = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor p_value = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor df = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    AT_DISPATCH_FLOATING_TYPES(
        input1_contig.scalar_type(),
        "paired_t_test_cpu",
        [&] {
            const scalar_t* data1_ptr = input1_contig.data_ptr<scalar_t>();
            const scalar_t* data2_ptr = input2_contig.data_ptr<scalar_t>();
            scalar_t* t_ptr = t_stat.data_ptr<scalar_t>();
            scalar_t* p_ptr = p_value.data_ptr<scalar_t>();
            scalar_t* df_ptr = df.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                // Thread-local scratch buffer for differences
                std::vector<scalar_t> diff_buffer(n_samples);

                for (int64_t b = begin; b < end; ++b) {
                    const scalar_t* d1 = data1_ptr + b * n_samples;
                    const scalar_t* d2 = data2_ptr + b * n_samples;

                    // Compute differences into thread-local buffer
                    for (int64_t i = 0; i < n_samples; ++i) {
                        diff_buffer[i] = d1[i] - d2[i];
                    }

                    // Use one_sample_t_test on differences with popmean=0
                    auto [t, p, d] = kernel::one_sample_t_test<scalar_t>(
                        diff_buffer.data(),
                        n_samples,
                        scalar_t(0),
                        alt
                    );
                    t_ptr[b] = t;
                    p_ptr[b] = p;
                    df_ptr[b] = d;
                }
            });
        }
    );

    return std::make_tuple(t_stat, p_value, df);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl(
        "paired_t_test",
        &torchscience::cpu::statistics::hypothesis_test::paired_t_test
    );
}
