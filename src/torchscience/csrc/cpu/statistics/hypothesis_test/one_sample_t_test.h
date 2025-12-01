// src/torchscience/csrc/cpu/statistics/hypothesis_test/one_sample_t_test.h
#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/one_sample_t_test.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * One-sample t-test CPU implementation.
 *
 * Tests if the mean of samples differs from a population mean.
 * Batches over all dimensions except the last (last dim = samples).
 *
 * @param input Input tensor where the last dimension contains samples
 * @param popmean Population mean to test against (null hypothesis)
 * @param alternative Alternative hypothesis: "two-sided", "less", or "greater"
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom) tensors
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> one_sample_t_test(
    const at::Tensor& input,
    double popmean,
    c10::string_view alternative
) {
    TORCH_CHECK(
        input.dim() >= 1,
        "one_sample_t_test: input must have at least 1 dimension, got ",
        input.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "one_sample_t_test: input must be floating-point, got ",
        input.scalar_type()
    );

    // Parse alternative hypothesis
    kernel::Alternative alt = kernel::parse_alternative(
        std::string(alternative).c_str()
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
        input_contig.scalar_type(),
        "one_sample_t_test_cpu",
        [&] {
            const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
            scalar_t* t_ptr = t_stat.data_ptr<scalar_t>();
            scalar_t* p_ptr = p_value.data_ptr<scalar_t>();
            scalar_t* df_ptr = df.data_ptr<scalar_t>();

            scalar_t popmean_scalar = static_cast<scalar_t>(popmean);

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    auto [t, p, d] = kernel::one_sample_t_test<scalar_t>(
                        data_ptr + b * n_samples,
                        n_samples,
                        popmean_scalar,
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
        "one_sample_t_test",
        &torchscience::cpu::statistics::hypothesis_test::one_sample_t_test
    );
}
