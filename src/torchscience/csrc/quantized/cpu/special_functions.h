#pragma once

#include "operators.h"

TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, m) {
    REGISTER_QUANTIZED_CPU_UNARY(m, gamma);
    REGISTER_QUANTIZED_CPU_BINARY(m, beta);
    REGISTER_QUANTIZED_CPU_BINARY(m, chebyshev_polynomial_t);
    REGISTER_QUANTIZED_CPU_TERNARY(m, incomplete_beta);
    REGISTER_QUANTIZED_CPU_QUATERNARY(m, hypergeometric_2_f_1);
}
