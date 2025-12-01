// src/torchscience/csrc/cpu/statistics/hypothesis_test/mann_whitney_u.h
#pragma once

#include <tuple>
#include <string>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/mann_whitney_u.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Mann-Whitney U test - CPU implementation.
 *
 * Tests whether two independent samples come from the same distribution.
 *
 * @param x First sample (1D tensor)
 * @param y Second sample (1D tensor)
 * @param alternative Alternative hypothesis: "two-sided", "less", "greater"
 * @return Tuple of (U-statistic, p-value) tensors (scalars)
 */
inline std::tuple<at::Tensor, at::Tensor> mann_whitney_u(
    const at::Tensor& x,
    const at::Tensor& y,
    c10::string_view alternative
) {
    TORCH_CHECK(
        x.dim() == 1,
        "mann_whitney_u: x must be 1-dimensional, got ", x.dim(), "D"
    );
    TORCH_CHECK(
        y.dim() == 1,
        "mann_whitney_u: y must be 1-dimensional, got ", y.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        "mann_whitney_u: x must be floating-point, got ", x.scalar_type()
    );
    TORCH_CHECK(
        x.scalar_type() == y.scalar_type(),
        "mann_whitney_u: x and y must have the same dtype"
    );
    TORCH_CHECK(
        !x.requires_grad() && !y.requires_grad(),
        "mann_whitney_u does not support gradients (rank-based test). "
        "Use with torch.no_grad() context."
    );

    // Parse alternative using shared utility
    kernel::Alternative alt = kernel::parse_alternative(std::string(alternative).c_str());

    // Make inputs contiguous
    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();

    int64_t n1 = x_contig.size(0);
    int64_t n2 = y_contig.size(0);

    // Create output tensors (scalars)
    auto options = x_contig.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);

    AT_DISPATCH_FLOATING_TYPES(
        x_contig.scalar_type(),
        "mann_whitney_u_cpu",
        [&] {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();

            auto [U, p] = kernel::mann_whitney_u<scalar_t>(
                x_ptr, n1, y_ptr, n2, alt
            );

            statistic.fill_(U);
            pvalue.fill_(p);
        }
    );

    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl(
        "mann_whitney_u",
        &torchscience::cpu::statistics::hypothesis_test::mann_whitney_u
    );
}
