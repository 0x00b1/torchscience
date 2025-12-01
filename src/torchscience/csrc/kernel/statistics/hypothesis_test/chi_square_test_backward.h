#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Backward pass for chi-square test.
 *
 * Computes gradients with respect to observed frequencies.
 *
 * chi2 = sum((O_i - E_i)^2 / E_i)
 * d(chi2)/d(O_i) = 2 * (O_i - E_i) / E_i
 *
 * For uniform expected (E_i = sum(O) / k):
 * d(chi2)/d(O_j) = 2 * (O_j - E) / E - 2 * sum((O_i - E) / E) / k
 *               = 2 * (O_j - E) / E - 2 * chi2_unnormalized / (k * E)
 *
 * @param grad_statistic Gradient with respect to chi-square statistic
 * @param observed Observed frequencies
 * @param expected Expected frequencies (nullptr for uniform)
 * @param grad_observed Output gradient (must be pre-allocated)
 * @param k Number of categories
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void chi_square_test_backward(
    T grad_statistic,
    const T* observed,
    const T* expected,
    T* grad_observed,
    int64_t k
) {
    if (k < 1) {
        return;
    }

    if (expected == nullptr) {
        // Uniform expected frequencies
        T total = T(0);
        for (int64_t i = 0; i < k; ++i) {
            total += observed[i];
        }
        T E = total / T(k);

        if (E <= T(0)) {
            for (int64_t i = 0; i < k; ++i) {
                grad_observed[i] = T(0);
            }
            return;
        }

        // For uniform expected, the gradient is more complex due to
        // E depending on all observed values.
        // chi2 = sum_i (O_i - E)^2 / E, where E = sum_j(O_j) / k
        //
        // d(chi2)/d(O_j) = d/d(O_j) [sum_i (O_i - E)^2 / E]
        //
        // Using quotient rule and chain rule:
        // Let S = sum_i (O_i - E)^2
        // chi2 = S / E
        // d(chi2)/d(O_j) = (dS/dO_j * E - S * dE/dO_j) / E^2
        //
        // dE/dO_j = 1/k
        // dS/dO_j = 2(O_j - E)(1 - 1/k) + sum_{i!=j} 2(O_i - E)(-1/k)
        //         = 2(O_j - E) - (2/k) * sum_i (O_i - E)
        //         = 2(O_j - E)  (since sum_i (O_i - E) = 0)
        //
        // So: d(chi2)/d(O_j) = [2(O_j - E) * E - S * (1/k)] / E^2
        //                    = 2(O_j - E)/E - S/(k * E^2)
        //                    = 2(O_j - E)/E - chi2/(k * E)

        T chi2 = T(0);
        for (int64_t i = 0; i < k; ++i) {
            T diff = observed[i] - E;
            chi2 += (diff * diff) / E;
        }

        for (int64_t i = 0; i < k; ++i) {
            T diff = observed[i] - E;
            grad_observed[i] = grad_statistic * (T(2) * diff / E - chi2 / (T(k) * E));
        }
    } else {
        // Explicit expected frequencies (not affected by observed)
        // chi2 = sum_i (O_i - E_i)^2 / E_i
        // d(chi2)/d(O_j) = 2 * (O_j - E_j) / E_j
        for (int64_t i = 0; i < k; ++i) {
            if (expected[i] <= T(0)) {
                grad_observed[i] = T(0);
            } else {
                T diff = observed[i] - expected[i];
                grad_observed[i] = grad_statistic * T(2) * diff / expected[i];
            }
        }
    }
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
