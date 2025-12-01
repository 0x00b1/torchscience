// src/torchscience/csrc/cpu/statistics/hypothesis_test/wilcoxon_signed_rank.h
#pragma once

#include <tuple>
#include <string>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/wilcoxon_signed_rank.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Parse zero_method string to enum.
 */
inline kernel::ZeroMethod parse_zero_method(const char* method) {
    if (std::strcmp(method, "wilcox") == 0) {
        return kernel::ZeroMethod::Wilcox;
    } else if (std::strcmp(method, "pratt") == 0) {
        return kernel::ZeroMethod::Pratt;
    } else if (std::strcmp(method, "zsplit") == 0) {
        return kernel::ZeroMethod::Zsplit;
    }
    TORCH_CHECK(false, "Invalid zero_method: ", method, ". Expected 'wilcox', 'pratt', or 'zsplit'.");
}

/**
 * Wilcoxon signed-rank test - CPU implementation.
 *
 * Tests whether the median of a sample (or differences) is zero.
 *
 * @param x First sample (1D tensor)
 * @param y Optional second sample (1D tensor). If provided, test on x - y.
 * @param alternative Alternative hypothesis: "two-sided", "less", "greater"
 * @param zero_method How to handle zero differences: "wilcox", "pratt", "zsplit"
 * @return Tuple of (W-statistic, p-value) tensors (scalars)
 */
inline std::tuple<at::Tensor, at::Tensor> wilcoxon_signed_rank(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& y,
    c10::string_view alternative,
    c10::string_view zero_method
) {
    TORCH_CHECK(
        x.dim() == 1,
        "wilcoxon_signed_rank: x must be 1-dimensional, got ", x.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        "wilcoxon_signed_rank: x must be floating-point, got ", x.scalar_type()
    );
    TORCH_CHECK(
        !x.requires_grad(),
        "wilcoxon_signed_rank does not support gradients (rank-based test). "
        "Use with torch.no_grad() context."
    );

    at::Tensor diff;

    if (y.has_value()) {
        const at::Tensor& y_val = y.value();
        TORCH_CHECK(
            y_val.dim() == 1,
            "wilcoxon_signed_rank: y must be 1-dimensional, got ", y_val.dim(), "D"
        );
        TORCH_CHECK(
            x.scalar_type() == y_val.scalar_type(),
            "wilcoxon_signed_rank: x and y must have the same dtype"
        );
        TORCH_CHECK(
            x.size(0) == y_val.size(0),
            "wilcoxon_signed_rank: x and y must have the same size"
        );
        TORCH_CHECK(
            !y_val.requires_grad(),
            "wilcoxon_signed_rank does not support gradients (rank-based test). "
            "Use with torch.no_grad() context."
        );
        diff = x - y_val;
    } else {
        diff = x;
    }

    diff = diff.contiguous();

    // Parse parameters
    kernel::Alternative alt = kernel::parse_alternative(std::string(alternative).c_str());
    kernel::ZeroMethod zm = parse_zero_method(std::string(zero_method).c_str());

    int64_t n = diff.size(0);

    // Create output tensors (scalars)
    auto options = x.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);

    AT_DISPATCH_FLOATING_TYPES(
        diff.scalar_type(),
        "wilcoxon_signed_rank_cpu",
        [&] {
            const scalar_t* d_ptr = diff.data_ptr<scalar_t>();

            auto [W, p] = kernel::wilcoxon_signed_rank<scalar_t>(
                d_ptr, n, alt, zm
            );

            statistic.fill_(W);
            pvalue.fill_(p);
        }
    );

    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl(
        "wilcoxon_signed_rank",
        &torchscience::cpu::statistics::hypothesis_test::wilcoxon_signed_rank
    );
}
