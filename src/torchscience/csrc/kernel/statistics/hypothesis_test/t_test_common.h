// src/torchscience/csrc/kernel/statistics/hypothesis_test/t_test_common.h
#pragma once

#include <cmath>
#include <cstring>
#include <limits>

#include <c10/macros/Macros.h>

#include "../../special_functions/incomplete_beta.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Alternative hypothesis type for t-tests.
 */
enum class Alternative {
    TWO_SIDED,
    LESS,
    GREATER
};

/**
 * Parse alternative hypothesis string.
 *
 * @param alt String: "two-sided", "less", or "greater"
 * @return Alternative enum value, defaults to TWO_SIDED for unknown strings
 */
C10_HOST_DEVICE C10_ALWAYS_INLINE
Alternative parse_alternative(const char* alt) {
    if (alt == nullptr) {
        return Alternative::TWO_SIDED;
    }

    // Compare strings manually for device compatibility
    if (std::strcmp(alt, "less") == 0) {
        return Alternative::LESS;
    }
    if (std::strcmp(alt, "greater") == 0) {
        return Alternative::GREATER;
    }

    return Alternative::TWO_SIDED;
}

/**
 * Compute p-value from t-statistic and degrees of freedom.
 *
 * Uses the incomplete beta function to compute the cumulative distribution
 * function of the t-distribution:
 *   CDF(t, df) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 0.5)  for t > 0
 *   CDF(t, df) = 0.5 * I_{df/(df+t^2)}(df/2, 0.5)      for t < 0
 *
 * @param t_stat The t-statistic
 * @param df Degrees of freedom
 * @param alternative The alternative hypothesis type
 * @return The p-value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T t_pvalue(T t_stat, T df, Alternative alternative) {
    using special_functions::incomplete_beta;

    if (df <= T(0) || std::isnan(t_stat) || std::isnan(df)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Compute x = df / (df + t^2) for the incomplete beta function
    T t2 = t_stat * t_stat;
    T x = df / (df + t2);

    // I_x(df/2, 0.5) gives the regularized incomplete beta
    T half = T(0.5);
    T a = df * half;
    T b = half;

    T ibeta = incomplete_beta(x, a, b);

    // CDF of t-distribution
    // For t >= 0: CDF = 1 - 0.5 * I_x(df/2, 0.5)
    // For t < 0:  CDF = 0.5 * I_x(df/2, 0.5)
    T cdf;
    if (t_stat >= T(0)) {
        cdf = T(1) - half * ibeta;
    } else {
        cdf = half * ibeta;
    }

    // Compute p-value based on alternative hypothesis
    T pvalue;
    switch (alternative) {
        case Alternative::LESS:
            // H1: mu < mu0, p-value = P(T <= t)
            pvalue = cdf;
            break;
        case Alternative::GREATER:
            // H1: mu > mu0, p-value = P(T >= t)
            pvalue = T(1) - cdf;
            break;
        case Alternative::TWO_SIDED:
        default:
            // H1: mu != mu0, p-value = 2 * min(CDF, 1 - CDF)
            pvalue = T(2) * std::min(cdf, T(1) - cdf);
            break;
    }

    return pvalue;
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
