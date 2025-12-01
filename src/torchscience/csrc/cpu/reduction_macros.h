// src/torchscience/csrc/cpu/reduction_macros.h
#pragma once

#include <numeric>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

// =============================================================================
// REDUCTION UTILITIES (internal to this header)
// =============================================================================

namespace torchscience::cpu::reduction_detail {

inline std::vector<int64_t> compute_reduction_shape(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim
) {
    std::vector<int64_t> output_shape;
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        if (keepdim) {
            output_shape.assign(ndim, 1);
        }
    } else {
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            TORCH_CHECK(pos_d >= 0 && pos_d < ndim,
                "Dimension out of range (expected to be in range of [",
                -ndim, ", ", ndim - 1, "], but got ", d, ")");
            reduce_dim[pos_d] = true;
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                if (keepdim) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input_sizes[i]);
            }
        }
    }

    return output_shape;
}

inline std::pair<int64_t, int64_t> compute_reduction_sizes(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim
) {
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        return {input.numel(), 1};
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    int64_t reduce_size = 1;
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            reduce_size *= input_sizes[i];
        } else {
            batch_size *= input_sizes[i];
        }
    }

    return {reduce_size, batch_size};
}

inline std::vector<int64_t> build_reduction_permutation(
    int64_t ndim,
    at::OptionalIntArrayRef dim
) {
    if (!dim.has_value() || dim->empty()) {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        return perm;
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    std::vector<int64_t> permutation;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!reduce_dim[i]) {
            permutation.push_back(i);
        }
    }
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            permutation.push_back(i);
        }
    }

    return permutation;
}

inline std::vector<int64_t> build_inverse_permutation(
    const std::vector<int64_t>& permutation
) {
    std::vector<int64_t> inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        inverse[permutation[i]] = static_cast<int64_t>(i);
    }
    return inverse;
}

enum class ReductionMode {
    LAST_DIM,
    ALL_DIMS
};

}  // namespace torchscience::cpu::reduction_detail

// =============================================================================
// DIM-BASED REDUCTION MACROS
// =============================================================================

/**
 * CPU macro for unary dim-based reduction operators (no extra params).
 *
 * Generates forward, backward, and backward_backward functions that:
 * - Handle arbitrary dim/keepdim combinations
 * - Permute input to make reduced dims contiguous
 * - Dispatch over floating and complex types
 * - Use parallel_for over batch dimensions
 * - Register with TORCH_LIBRARY_IMPL
 *
 * Kernel interface (in namespace torchscience::kernel::NS):
 *   template<T> T name(const T* data, int64_t n)
 *   template<T> void name_backward(T grad_out, const T* data, int64_t n, T* grad_input)
 *   template<T> void name_backward_backward(const T* gg_input, T grad_out, const T* data, int64_t n, T& gg_output, T* new_grad_input)
 *
 * @param NS Namespace suffix (e.g., statistics::descriptive)
 * @param name Operator name (e.g., kurtosis)
 * @param arg Tensor argument name (e.g., input)
 *
 * For operators with extra parameters (like bool fisher, bool bias), use
 * TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR_EX instead.
 */
#define TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR(NS, name, arg) \
    TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR_EX(NS, name, arg, , )

/**
 * CPU macro for unary dim-based reduction operators with extra parameters.
 *
 * @param NS Namespace suffix (e.g., statistics::descriptive)
 * @param name Operator name (e.g., kurtosis)
 * @param arg Tensor argument name (e.g., input)
 * @param EXTRA_PARAMS Extra param declarations with leading comma, or empty
 * @param EXTRA_ARGS Extra param names with leading comma, or empty
 *
 * Example:
 *   TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR_EX(
 *       statistics::descriptive, kurtosis, input,
 *       TSCI_EXTRA(bool fisher, bool bias),  // EXTRA_PARAMS
 *       TSCI_EXTRA(fisher, bias)             // EXTRA_ARGS
 *   )
 *
 * Use TSCI_EXTRA(...) wrapper for extra params, or TSCI_NO_EXTRA for none.
 */
#define TSCI_EXTRA(...) , __VA_ARGS__
#define TSCI_NO_EXTRA
#define TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR_EX(NS, name, arg, EXTRA_PARAMS, EXTRA_ARGS)       \
namespace torchscience::cpu::NS {                                               \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    TORCH_CHECK(arg.numel() > 0, #name ": input tensor must be non-empty");     \
                                                                                \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    auto output_shape = compute_reduction_shape(arg, dim, keepdim);             \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg, dim);         \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
                                                                                \
    auto output_dtype = at::isComplexType(arg.scalar_type())                    \
        ? c10::toRealValueType(arg.scalar_type())                               \
        : arg.scalar_type();                                                    \
                                                                                \
    auto options = arg##_contig.options().dtype(output_dtype);                  \
    at::Tensor output = output_shape.empty()                                    \
        ? at::empty({}, options)                                                \
        : at::empty(output_shape, options);                                     \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_cpu_all",                                                   \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t result = kernel::NS::name<scalar_t>(                   \
                    data_ptr, arg##_contig.numel()                              \
                    EXTRA_ARGS                                                  \
                );                                                              \
                output.fill_(result);                                           \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    }                                                                           \
                                                                                \
    auto permutation = build_reduction_permutation(arg.dim(), dim);             \
    auto permuted = arg##_contig.permute(permutation).contiguous();             \
    auto permuted_view = permuted.view({batch_size, reduce_size});              \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
        at::kBFloat16, at::kHalf,                                               \
        arg##_contig.scalar_type(),                                             \
        #name "_cpu_dim",                                                       \
        [&]() {                                                                 \
            const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();      \
            scalar_t* output_ptr = output.data_ptr<scalar_t>();                 \
                                                                                \
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                for (int64_t b = begin; b < end; ++b) {                         \
                    output_ptr[b] = kernel::NS::name<scalar_t>(                 \
                        data_ptr + b * reduce_size,                             \
                        reduce_size                                             \
                        EXTRA_ARGS                                              \
                    );                                                          \
                }                                                               \
            });                                                                 \
        }                                                                       \
    );                                                                          \
                                                                                \
    return output;                                                              \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    at::Tensor grad_input = at::zeros_like(arg);                                \
    auto arg##_contig = arg.contiguous();                                       \
    auto grad_input_contig = grad_input.contiguous();                           \
                                                                                \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg, dim);         \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_cpu_all",                                          \
            [&]() {                                                             \
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t* grad_ptr = grad_input_contig.data_ptr<scalar_t>();    \
                                                                                \
                kernel::NS::name##_backward<scalar_t>(                          \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    EXTRA_ARGS                                                  \
                    , grad_ptr                                                  \
                );                                                              \
            }                                                                   \
        );                                                                      \
        return grad_input_contig;                                               \
    }                                                                           \
                                                                                \
    auto permutation = build_reduction_permutation(arg.dim(), dim);             \
    auto permuted = arg##_contig.permute(permutation).contiguous();             \
    auto permuted_view = permuted.view({batch_size, reduce_size});              \
                                                                                \
    at::Tensor grad_output_expanded;                                            \
    if (keepdim) {                                                              \
        grad_output_expanded = grad_output.contiguous().view({batch_size});     \
    } else {                                                                    \
        int64_t ndim = arg.dim();                                               \
        std::vector<bool> reduce_dim(ndim, false);                              \
        for (int64_t d : *dim) {                                                \
            int64_t pos_d = d >= 0 ? d : d + ndim;                              \
            reduce_dim[pos_d] = true;                                           \
        }                                                                       \
        at::Tensor temp = grad_output;                                          \
        for (int64_t i = 0; i < ndim; ++i) {                                    \
            if (reduce_dim[i]) {                                                \
                temp = temp.unsqueeze(i);                                       \
            }                                                                   \
        }                                                                       \
        grad_output_expanded = temp.contiguous().view({batch_size});            \
    }                                                                           \
                                                                                \
    at::Tensor grad_permuted = at::zeros({batch_size, reduce_size}, arg.options());\
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
        at::kBFloat16, at::kHalf,                                               \
        arg##_contig.scalar_type(),                                             \
        #name "_backward_cpu_dim",                                              \
        [&]() {                                                                 \
            const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();      \
            const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();\
            scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();            \
                                                                                \
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                for (int64_t b = begin; b < end; ++b) {                         \
                    kernel::NS::name##_backward<scalar_t>(                      \
                        grad_out_ptr[b],                                        \
                        data_ptr + b * reduce_size,                             \
                        reduce_size                                             \
                        EXTRA_ARGS                                              \
                        , grad_ptr + b * reduce_size                            \
                    );                                                          \
                }                                                               \
            });                                                                 \
        }                                                                       \
    );                                                                          \
                                                                                \
    auto inverse_perm = build_inverse_permutation(permutation);                 \
    return grad_permuted.view(permuted.sizes())                                 \
        .permute(inverse_perm)                                                  \
        .contiguous();                                                          \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    at::Tensor grad_grad_output = at::zeros_like(grad_output);                  \
    at::Tensor new_grad_input = at::zeros_like(arg);                            \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
    auto grad_grad_input_contig = grad_grad_input.contiguous();                 \
                                                                                \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg, dim);         \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_backward_cpu_all",                                 \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* gg_input_ptr = grad_grad_input_contig.data_ptr<scalar_t>();\
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                                                                                \
                scalar_t gg_output;                                             \
                                                                                \
                kernel::NS::name##_backward_backward<scalar_t>(                 \
                    gg_input_ptr,                                               \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    EXTRA_ARGS                                                  \
                    , gg_output                                                 \
                    , new_grad_input.data_ptr<scalar_t>()                       \
                );                                                              \
                                                                                \
                grad_grad_output.fill_(gg_output);                              \
            }                                                                   \
        );                                                                      \
                                                                                \
        return std::make_tuple(grad_grad_output, new_grad_input);               \
    }                                                                           \
                                                                                \
    auto permutation = build_reduction_permutation(arg.dim(), dim);             \
    auto permuted_input = arg##_contig.permute(permutation).contiguous();       \
    auto permuted_gg_input = grad_grad_input_contig.permute(permutation).contiguous();\
    auto permuted_input_view = permuted_input.view({batch_size, reduce_size});  \
    auto permuted_gg_input_view = permuted_gg_input.view({batch_size, reduce_size});\
                                                                                \
    at::Tensor grad_output_expanded;                                            \
    if (keepdim) {                                                              \
        grad_output_expanded = grad_output.contiguous().view({batch_size});     \
    } else {                                                                    \
        int64_t ndim = arg.dim();                                               \
        std::vector<bool> reduce_dim(ndim, false);                              \
        for (int64_t d : *dim) {                                                \
            int64_t pos_d = d >= 0 ? d : d + ndim;                              \
            reduce_dim[pos_d] = true;                                           \
        }                                                                       \
        at::Tensor temp = grad_output;                                          \
        for (int64_t i = 0; i < ndim; ++i) {                                    \
            if (reduce_dim[i]) {                                                \
                temp = temp.unsqueeze(i);                                       \
            }                                                                   \
        }                                                                       \
        grad_output_expanded = temp.contiguous().view({batch_size});            \
    }                                                                           \
                                                                                \
    at::Tensor new_grad_permuted = at::zeros({batch_size, reduce_size}, arg.options());\
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
        at::kBFloat16, at::kHalf,                                               \
        arg##_contig.scalar_type(),                                             \
        #name "_backward_backward_cpu_dim",                                     \
        [&]() {                                                                 \
            const scalar_t* data_ptr = permuted_input_view.data_ptr<scalar_t>();\
            const scalar_t* gg_input_ptr = permuted_gg_input_view.data_ptr<scalar_t>();\
            const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();\
            scalar_t* gg_output_ptr = grad_grad_output.data_ptr<scalar_t>();    \
            scalar_t* new_grad_ptr = new_grad_permuted.data_ptr<scalar_t>();    \
                                                                                \
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                for (int64_t b = begin; b < end; ++b) {                         \
                    scalar_t gg_out;                                            \
                    kernel::NS::name##_backward_backward<scalar_t>(             \
                        gg_input_ptr + b * reduce_size,                         \
                        grad_out_ptr[b],                                        \
                        data_ptr + b * reduce_size,                             \
                        reduce_size                                             \
                        EXTRA_ARGS                                              \
                        , gg_out                                                \
                        , new_grad_ptr + b * reduce_size                        \
                    );                                                          \
                    gg_output_ptr[b] = gg_out;                                  \
                }                                                               \
            });                                                                 \
        }                                                                       \
    );                                                                          \
                                                                                \
    auto inverse_perm = build_inverse_permutation(permutation);                 \
    new_grad_input = new_grad_permuted.view(permuted_input.sizes())             \
        .permute(inverse_perm)                                                  \
        .contiguous();                                                          \
                                                                                \
    return std::make_tuple(grad_grad_output, new_grad_input);                   \
}                                                                               \
                                                                                \
} /* namespace torchscience::cpu::NS */                                         \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, m) {                                      \
    m.impl(#name, &torchscience::cpu::NS::name);                                \
    m.impl(#name "_backward", &torchscience::cpu::NS::name##_backward);         \
    m.impl(#name "_backward_backward", &torchscience::cpu::NS::name##_backward_backward);\
}

// =============================================================================
// FIXED REDUCTION MACROS
// =============================================================================

/**
 * CPU macro for unary fixed reduction operators (no extra params).
 *
 * @param NS Namespace suffix
 * @param name Operator name
 * @param MODE ReductionMode::LAST_DIM or ReductionMode::ALL_DIMS
 * @param arg Tensor argument name
 *
 * For operators with extra parameters, use
 * TORCHSCIENCE_CPU_FIXED_REDUCTION_UNARY_OPERATOR_EX instead.
 */
#define TORCHSCIENCE_CPU_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, MODE, arg) \
    TORCHSCIENCE_CPU_FIXED_REDUCTION_UNARY_OPERATOR_EX(NS, name, MODE, arg, , )

/**
 * CPU macro for unary fixed reduction operators with extra parameters.
 *
 * @param NS Namespace suffix
 * @param name Operator name
 * @param MODE ReductionMode::LAST_DIM or ReductionMode::ALL_DIMS
 * @param arg Tensor argument name
 * @param EXTRA_PARAMS Extra param declarations with leading comma, or empty
 * @param EXTRA_ARGS Extra param names with leading comma, or empty
 *
 * Example:
 *   TORCHSCIENCE_CPU_FIXED_REDUCTION_UNARY_OPERATOR_EX(
 *       NS, name, ReductionMode::ALL_DIMS, input,
 *       TSCI_EXTRA(bool normalize),
 *       TSCI_EXTRA(normalize)
 *   )
 */
#define TORCHSCIENCE_CPU_FIXED_REDUCTION_UNARY_OPERATOR_EX(NS, name, MODE, arg, EXTRA_PARAMS, EXTRA_ARGS)\
namespace torchscience::cpu::NS {                                               \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        TORCH_CHECK(arg.numel() > 0, #name ": input tensor must be non-empty"); \
        auto output = at::empty({}, arg##_contig.options());                    \
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_cpu_all",                                                   \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t result = kernel::NS::name<scalar_t>(                   \
                    data_ptr, arg##_contig.numel()                              \
                    EXTRA_ARGS                                                  \
                );                                                              \
                output.fill_(result);                                           \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    } else {                                                                    \
        TORCH_CHECK(arg.dim() >= 1, #name ": input must have at least 1 dimension");\
                                                                                \
        int64_t reduce_size = arg##_contig.size(-1);                            \
        int64_t batch_size = arg##_contig.numel() / reduce_size;                \
                                                                                \
        auto output_shape = arg##_contig.sizes().vec();                         \
        output_shape.pop_back();                                                \
        auto output = output_shape.empty()                                      \
            ? at::empty({}, arg##_contig.options())                             \
            : at::empty(output_shape, arg##_contig.options());                  \
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_cpu_last",                                                  \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t* output_ptr = output.data_ptr<scalar_t>();             \
                                                                                \
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                    for (int64_t b = begin; b < end; ++b) {                     \
                        output_ptr[b] = kernel::NS::name<scalar_t>(             \
                            data_ptr + b * reduce_size,                         \
                            reduce_size                                         \
                            EXTRA_ARGS                                          \
                        );                                                      \
                    }                                                           \
                });                                                             \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    }                                                                           \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    at::Tensor grad_input = at::zeros_like(arg);                                \
    auto arg##_contig = arg.contiguous();                                       \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_cpu_all",                                          \
            [&]() {                                                             \
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t* grad_ptr = grad_input.data_ptr<scalar_t>();           \
                                                                                \
                kernel::NS::name##_backward<scalar_t>(                          \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    EXTRA_ARGS                                                  \
                    , grad_ptr                                                  \
                );                                                              \
            }                                                                   \
        );                                                                      \
    } else {                                                                    \
        int64_t reduce_size = arg##_contig.size(-1);                            \
        int64_t batch_size = arg##_contig.numel() / reduce_size;                \
                                                                                \
        auto grad_output_flat = grad_output.contiguous().view({batch_size});    \
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_cpu_last",                                         \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();\
                scalar_t* grad_ptr = grad_input.data_ptr<scalar_t>();           \
                                                                                \
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                    for (int64_t b = begin; b < end; ++b) {                     \
                        kernel::NS::name##_backward<scalar_t>(                  \
                            grad_out_ptr[b],                                    \
                            data_ptr + b * reduce_size,                         \
                            reduce_size                                         \
                            EXTRA_ARGS                                          \
                            , grad_ptr + b * reduce_size                        \
                        );                                                      \
                    }                                                           \
                });                                                             \
            }                                                                   \
        );                                                                      \
    }                                                                           \
    return grad_input;                                                          \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    at::Tensor grad_grad_output = at::zeros_like(grad_output);                  \
    at::Tensor new_grad_input = at::zeros_like(arg);                            \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
    auto grad_grad_input_contig = grad_grad_input.contiguous();                 \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_backward_cpu_all",                                 \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* gg_input_ptr = grad_grad_input_contig.data_ptr<scalar_t>();\
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                                                                                \
                scalar_t gg_output;                                             \
                                                                                \
                kernel::NS::name##_backward_backward<scalar_t>(                 \
                    gg_input_ptr,                                               \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    EXTRA_ARGS                                                  \
                    , gg_output                                                 \
                    , new_grad_input.data_ptr<scalar_t>()                       \
                );                                                              \
                                                                                \
                grad_grad_output.fill_(gg_output);                              \
            }                                                                   \
        );                                                                      \
    } else {                                                                    \
        int64_t reduce_size = arg##_contig.size(-1);                            \
        int64_t batch_size = arg##_contig.numel() / reduce_size;                \
                                                                                \
        auto grad_output_flat = grad_output.contiguous().view({batch_size});    \
        auto gg_input_view = grad_grad_input_contig.view({batch_size, reduce_size});\
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_backward_cpu_last",                                \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* gg_input_ptr = gg_input_view.data_ptr<scalar_t>();\
                const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();\
                scalar_t* gg_output_ptr = grad_grad_output.data_ptr<scalar_t>();\
                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();   \
                                                                                \
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                    for (int64_t b = begin; b < end; ++b) {                     \
                        scalar_t gg_out;                                        \
                        kernel::NS::name##_backward_backward<scalar_t>(         \
                            gg_input_ptr + b * reduce_size,                     \
                            grad_out_ptr[b],                                    \
                            data_ptr + b * reduce_size,                         \
                            reduce_size                                         \
                            EXTRA_ARGS                                          \
                            , gg_out                                            \
                            , new_grad_ptr + b * reduce_size                    \
                        );                                                      \
                        gg_output_ptr[b] = gg_out;                              \
                    }                                                           \
                });                                                             \
            }                                                                   \
        );                                                                      \
    }                                                                           \
    return std::make_tuple(grad_grad_output, new_grad_input);                   \
}                                                                               \
                                                                                \
} /* namespace torchscience::cpu::NS */                                         \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, m) {                                      \
    m.impl(#name, &torchscience::cpu::NS::name);                                \
    m.impl(#name "_backward", &torchscience::cpu::NS::name##_backward);         \
    m.impl(#name "_backward_backward", &torchscience::cpu::NS::name##_backward_backward);\
}
