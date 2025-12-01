// src/torchscience/csrc/autograd/statistics/descriptive/kurtosis.h
#pragma once

#include "../../reduction_macros.h"

// Use the reduction macro with extra parameters (fisher, bias)
TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(
    statistics::descriptive,
    kurtosis,
    Kurtosis,
    input,
    TSCI_EXTRA(bool fisher, bool bias),
    TSCI_EXTRA(fisher, bias),
    TSCI_TYPES(bool, bool),
    TSCI_SAVE(ctx->saved_data["fisher"] = fisher; ctx->saved_data["bias"] = bias;),
    TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool(); bool bias = ctx->saved_data["bias"].toBool();),
    TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())
)
