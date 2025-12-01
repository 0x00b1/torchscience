// src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h
#pragma once

#include "../../../kernel/statistics/descriptive/kurtosis.h"
#include "../../reduction_macros.h"

// Use the reduction macro with extra parameters (fisher, bias)
TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR_EX(
    statistics::descriptive,
    kurtosis,
    input,
    TSCI_EXTRA(bool fisher, bool bias),
    TSCI_EXTRA(fisher, bias)
)
