#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::quantized::cpu {

// ============================================================================
// Quantized CPU operators dequantize, delegate to dense ops, then requantize
// ============================================================================

namespace detail {

inline at::Tensor requantize(
    const at::Tensor& result,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype
) {
    return at::quantize_per_tensor(result, scale, zero_point, dtype);
}

}  // namespace detail

// ============================================================================
// QuantizedCpuUnaryOperator
// ============================================================================

struct QuantizedCpuUnaryOperator {
    static at::Tensor forward(
        const at::Tensor& input,
        const char* schema_name
    ) {
        TORCH_CHECK(input.is_quantized(), "expects quantized tensor");

        at::Tensor dequantized = input.dequantize();

        at::Tensor result = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(dequantized);

        return detail::requantize(
            result, input.q_scale(), input.q_zero_point(), input.scalar_type()
        );
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_quantized(), "expects quantized gradient");
        TORCH_CHECK(input.is_quantized(), "expects quantized input");

        at::Tensor result = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(grad_output.dequantize(), input.dequantize());

        return detail::requantize(
            result, input.q_scale(), input.q_zero_point(), input.scalar_type()
        );
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const char* schema_name
    ) {
        if (!grad_grad_input.defined()) {
            return {};
        }

        TORCH_CHECK(grad_grad_input.is_quantized(), "expects quantized grad_grad");
        TORCH_CHECK(grad_output.is_quantized(), "expects quantized gradient");
        TORCH_CHECK(input.is_quantized(), "expects quantized input");

        auto [grad_grad_out, grad_input] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_grad_input.dequantize(), grad_output.dequantize(),
                  input.dequantize());

        at::Tensor quantized_grad_grad_out;
        at::Tensor quantized_grad_input;

        if (grad_grad_out.defined()) {
            quantized_grad_grad_out = detail::requantize(
                grad_grad_out, grad_output.q_scale(),
                grad_output.q_zero_point(), grad_output.scalar_type()
            );
        }

        if (grad_input.defined()) {
            quantized_grad_input = detail::requantize(
                grad_input, input.q_scale(),
                input.q_zero_point(), input.scalar_type()
            );
        }

        return {quantized_grad_grad_out, quantized_grad_input};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name,
        const char* schema_name,
        const char* schema_backward_name,
        const char* schema_backward_backward_name
    ) {
        module.impl(name, [schema_name](const at::Tensor& input) {
            return forward(input, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output, const at::Tensor& input
        ) {
            return backward(grad_output, input, schema_backward_name);
        });
        module.impl(backward_backward_name, [schema_backward_backward_name](
            const at::Tensor& grad_grad_input,
            const at::Tensor& grad_output,
            const at::Tensor& input
        ) {
            return backward_backward(
                grad_grad_input, grad_output, input, schema_backward_backward_name
            );
        });
    }
};

#define REGISTER_QUANTIZED_CPU_UNARY(module, name) \
    ::torchscience::quantized::cpu::QuantizedCpuUnaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward", \
        "torchscience::" #name "_backward_backward")

// ============================================================================
// QuantizedCpuBinaryOperator
// ============================================================================

struct QuantizedCpuBinaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const char* schema_name
    ) {
        TORCH_CHECK(input1.is_quantized() && input2.is_quantized(),
                    "expects quantized tensors");

        at::Tensor result = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(input1.dequantize(), input2.dequantize());

        return detail::requantize(
            result, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
        );
    }

    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_quantized(), "expects quantized gradient");
        TORCH_CHECK(input1.is_quantized() && input2.is_quantized(),
                    "expects quantized inputs");

        auto [grad1, grad2] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output.dequantize(), input1.dequantize(), input2.dequantize());

        at::Tensor quant_grad1, quant_grad2;

        if (grad1.defined()) {
            quant_grad1 = detail::requantize(
                grad1, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
            );
        }

        if (grad2.defined()) {
            quant_grad2 = detail::requantize(
                grad2, input2.q_scale(), input2.q_zero_point(), input2.scalar_type()
            );
        }

        return {quant_grad1, quant_grad2};
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& gg_input1,
        const at::Tensor& gg_input2,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const char* schema_name
    ) {
        at::Tensor gg1_dequant = gg_input1.defined() && gg_input1.is_quantized()
            ? gg_input1.dequantize() : gg_input1;
        at::Tensor gg2_dequant = gg_input2.defined() && gg_input2.is_quantized()
            ? gg_input2.dequantize() : gg_input2;

        auto [gg_out, grad1, grad2] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(gg1_dequant, gg2_dequant, grad_output.dequantize(),
                  input1.dequantize(), input2.dequantize());

        at::Tensor quant_gg_out, quant_grad1, quant_grad2;

        if (gg_out.defined()) {
            quant_gg_out = detail::requantize(
                gg_out, grad_output.q_scale(),
                grad_output.q_zero_point(), grad_output.scalar_type()
            );
        }

        if (grad1.defined()) {
            quant_grad1 = detail::requantize(
                grad1, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
            );
        }

        if (grad2.defined()) {
            quant_grad2 = detail::requantize(
                grad2, input2.q_scale(), input2.q_zero_point(), input2.scalar_type()
            );
        }

        return {quant_gg_out, quant_grad1, quant_grad2};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name,
        const char* schema_name,
        const char* schema_backward_name,
        const char* schema_backward_backward_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1, const at::Tensor& input2
        ) {
            return forward(input1, input2, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2
        ) {
            return backward(grad_output, input1, input2, schema_backward_name);
        });
        module.impl(backward_backward_name, [schema_backward_backward_name](
            const at::Tensor& gg_input1,
            const at::Tensor& gg_input2,
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2
        ) {
            return backward_backward(
                gg_input1, gg_input2, grad_output, input1, input2,
                schema_backward_backward_name
            );
        });
    }
};

#define REGISTER_QUANTIZED_CPU_BINARY(module, name) \
    ::torchscience::quantized::cpu::QuantizedCpuBinaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward", \
        "torchscience::" #name "_backward_backward")

// ============================================================================
// QuantizedCpuTernaryOperator
// ============================================================================

struct QuantizedCpuTernaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const char* schema_name
    ) {
        TORCH_CHECK(
            input1.is_quantized() && input2.is_quantized() && input3.is_quantized(),
            "expects quantized tensors"
        );

        at::Tensor result = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(input1.dequantize(), input2.dequantize(), input3.dequantize());

        return detail::requantize(
            result, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
        );
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_quantized(), "expects quantized gradient");
        TORCH_CHECK(
            input1.is_quantized() && input2.is_quantized() && input3.is_quantized(),
            "expects quantized inputs"
        );

        auto [grad1, grad2, grad3] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output.dequantize(), input1.dequantize(),
                  input2.dequantize(), input3.dequantize());

        at::Tensor quant_grad1, quant_grad2, quant_grad3;

        if (grad1.defined()) {
            quant_grad1 = detail::requantize(
                grad1, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
            );
        }

        if (grad2.defined()) {
            quant_grad2 = detail::requantize(
                grad2, input2.q_scale(), input2.q_zero_point(), input2.scalar_type()
            );
        }

        if (grad3.defined()) {
            quant_grad3 = detail::requantize(
                grad3, input3.q_scale(), input3.q_zero_point(), input3.scalar_type()
            );
        }

        return {quant_grad1, quant_grad2, quant_grad3};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* schema_name,
        const char* schema_backward_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3
        ) {
            return forward(input1, input2, input3, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3
        ) {
            return backward(grad_output, input1, input2, input3, schema_backward_name);
        });
    }
};

#define REGISTER_QUANTIZED_CPU_TERNARY(module, name) \
    ::torchscience::quantized::cpu::QuantizedCpuTernaryOperator::register_all( \
        module, #name, #name "_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward")

// ============================================================================
// QuantizedCpuQuaternaryOperator
// ============================================================================

struct QuantizedCpuQuaternaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4,
        const char* schema_name
    ) {
        TORCH_CHECK(
            input1.is_quantized() && input2.is_quantized() &&
            input3.is_quantized() && input4.is_quantized(),
            "expects quantized tensors"
        );

        at::Tensor result = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(input1.dequantize(), input2.dequantize(),
                  input3.dequantize(), input4.dequantize());

        return detail::requantize(
            result, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
        );
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4,
        const char* schema_name
    ) {
        TORCH_CHECK(grad_output.is_quantized(), "expects quantized gradient");
        TORCH_CHECK(
            input1.is_quantized() && input2.is_quantized() &&
            input3.is_quantized() && input4.is_quantized(),
            "expects quantized inputs"
        );

        auto [grad1, grad2, grad3, grad4] = c10::Dispatcher::singleton()
            .findSchemaOrThrow(schema_name, "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output.dequantize(), input1.dequantize(),
                  input2.dequantize(), input3.dequantize(), input4.dequantize());

        at::Tensor quant_grad1, quant_grad2, quant_grad3, quant_grad4;

        if (grad1.defined()) {
            quant_grad1 = detail::requantize(
                grad1, input1.q_scale(), input1.q_zero_point(), input1.scalar_type()
            );
        }

        if (grad2.defined()) {
            quant_grad2 = detail::requantize(
                grad2, input2.q_scale(), input2.q_zero_point(), input2.scalar_type()
            );
        }

        if (grad3.defined()) {
            quant_grad3 = detail::requantize(
                grad3, input3.q_scale(), input3.q_zero_point(), input3.scalar_type()
            );
        }

        if (grad4.defined()) {
            quant_grad4 = detail::requantize(
                grad4, input4.q_scale(), input4.q_zero_point(), input4.scalar_type()
            );
        }

        return {quant_grad1, quant_grad2, quant_grad3, quant_grad4};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* schema_name,
        const char* schema_backward_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4
        ) {
            return forward(input1, input2, input3, input4, schema_name);
        });
        module.impl(backward_name, [schema_backward_name](
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4
        ) {
            return backward(
                grad_output, input1, input2, input3, input4, schema_backward_name
            );
        });
    }
};

#define REGISTER_QUANTIZED_CPU_QUATERNARY(module, name) \
    ::torchscience::quantized::cpu::QuantizedCpuQuaternaryOperator::register_all( \
        module, #name, #name "_backward", \
        "torchscience::" #name, \
        "torchscience::" #name "_backward")

}  // namespace torchscience::quantized::cpu
