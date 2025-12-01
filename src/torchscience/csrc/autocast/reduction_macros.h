// src/torchscience/csrc/autocast/reduction_macros.h
#pragma once

#include <tuple>

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// =============================================================================
// HELPER MACROS
// =============================================================================

// Use these to wrap extra parameters for _EX macros
#ifndef TSCI_EXTRA
#define TSCI_EXTRA(...) , __VA_ARGS__
#endif
#ifndef TSCI_NO_EXTRA
#define TSCI_NO_EXTRA
#endif
#ifndef TSCI_TYPES
#define TSCI_TYPES(...) , __VA_ARGS__
#endif
#ifndef TSCI_NO_TYPES
#define TSCI_NO_TYPES
#endif

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autocast)
// =============================================================================

/**
 * Autocast macro for unary dim-based reduction operators (no extra params).
 */
#define TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR(NS, name, arg) \
    TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR_EX(NS, name, arg, , , )

/**
 * Autocast macro for unary dim-based reduction operators with extra parameters.
 *
 * @param NS Namespace suffix
 * @param name Operator name
 * @param arg Tensor argument name
 * @param EXTRA_PARAMS Param declarations with leading comma: TSCI_EXTRA(bool fisher, bool bias)
 * @param EXTRA_ARGS Param names with leading comma: TSCI_EXTRA(fisher, bias)
 * @param EXTRA_TYPES Type list with leading comma: TSCI_TYPES(bool, bool)
 *
 * Example:
 *   TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR_EX(
 *       statistics::descriptive, kurtosis, input,
 *       TSCI_EXTRA(bool fisher, bool bias),
 *       TSCI_EXTRA(fisher, bias),
 *       TSCI_TYPES(bool, bool)
 *   )
 */
#define TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR_EX(NS, name, arg, EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES)\
namespace torchscience::autocast::NS {                                          \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name, "")                          \
        .typed<at::Tensor(                                                      \
            const at::Tensor&, at::OptionalIntArrayRef, bool                    \
            EXTRA_TYPES                                                         \
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, arg),                         \
            dim,                                                                \
            keepdim                                                             \
            EXTRA_ARGS                                                          \
        );                                                                      \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward", "")              \
        .typed<at::Tensor(                                                      \
            const at::Tensor&, const at::Tensor&,                               \
            at::OptionalIntArrayRef, bool                                       \
            EXTRA_TYPES                                                         \
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg),                         \
            dim,                                                                \
            keepdim                                                             \
            EXTRA_ARGS                                                          \
        );                                                                      \
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
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward_backward", "")     \
        .typed<std::tuple<at::Tensor, at::Tensor>(                              \
            const at::Tensor&, const at::Tensor&, const at::Tensor&,            \
            at::OptionalIntArrayRef, bool                                       \
            EXTRA_TYPES                                                         \
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_grad_input),             \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg),                         \
            dim,                                                                \
            keepdim                                                             \
            EXTRA_ARGS                                                          \
        );                                                                      \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {                                 \
    m.impl(#name, &torchscience::autocast::NS::name);                           \
    m.impl(#name "_backward", &torchscience::autocast::NS::name##_backward);    \
    m.impl(#name "_backward_backward", &torchscience::autocast::NS::name##_backward_backward);\
}

// =============================================================================
// FIXED REDUCTION MACROS (Autocast)
// =============================================================================

/**
 * Autocast macro for unary fixed reduction operators (no extra params).
 */
#define TORCHSCIENCE_AUTOCAST_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, arg) \
    TORCHSCIENCE_AUTOCAST_FIXED_REDUCTION_UNARY_OPERATOR_EX(NS, name, arg, , , )

/**
 * Autocast macro for unary fixed reduction operators with extra parameters.
 */
#define TORCHSCIENCE_AUTOCAST_FIXED_REDUCTION_UNARY_OPERATOR_EX(NS, name, arg, EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES)\
namespace torchscience::autocast::NS {                                          \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name, "")                          \
        .typed<at::Tensor(const at::Tensor&                                     \
            EXTRA_TYPES                                                         \
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, arg)                          \
            EXTRA_ARGS                                                          \
        );                                                                      \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward", "")              \
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&                  \
            EXTRA_TYPES                                                         \
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg)                          \
            EXTRA_ARGS                                                          \
        );                                                                      \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward_backward", "")     \
        .typed<std::tuple<at::Tensor, at::Tensor>(                              \
            const at::Tensor&, const at::Tensor&, const at::Tensor&             \
            EXTRA_TYPES                                                         \
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_grad_input),             \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg)                          \
            EXTRA_ARGS                                                          \
        );                                                                      \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {                                 \
    m.impl(#name, &torchscience::autocast::NS::name);                           \
    m.impl(#name "_backward", &torchscience::autocast::NS::name##_backward);    \
    m.impl(#name "_backward_backward", &torchscience::autocast::NS::name##_backward_backward);\
}
