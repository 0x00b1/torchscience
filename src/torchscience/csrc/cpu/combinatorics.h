#pragma once

#include "macros.h"

#include "../kernel/combinatorics/binomial_coefficient.h"
#include "../kernel/combinatorics/binomial_coefficient_backward.h"
#include "../kernel/combinatorics/binomial_coefficient_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(binomial_coefficient, n, k)
