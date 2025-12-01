// src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h
#pragma once

#include "../../reduction_macros.h"

// Use the reduction macro with extra parameters (fisher, bias)
// Note: Extra params are unused in meta - just for shape inference
TORCHSCIENCE_META_DIM_REDUCTION_UNARY_OPERATOR_EX(
    statistics::descriptive,
    kurtosis,
    input,
    TSCI_EXTRA_UNUSED(bool fisher, bool bias),
    TSCI_EXTRA(fisher, bias)
)
