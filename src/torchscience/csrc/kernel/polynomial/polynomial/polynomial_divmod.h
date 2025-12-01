#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial division (long division algorithm)
// Computes quotient Q and remainder R such that: p = q * Q + R
// where deg(R) < deg(q)
//
// Parameters:
//   quotient: output array of size N - M + 1 (deg_p - deg_q + 1)
//   remainder: output array of size M (deg_q + 1, but only first M-1 are meaningful if M > 1)
//   p: dividend polynomial coefficients, size N (deg_p + 1)
//   q: divisor polynomial coefficients, size M (deg_q + 1)
//   N: number of coefficients in p
//   M: number of coefficients in q
//
// Preconditions:
//   - N >= M (deg_p >= deg_q)
//   - q[M-1] != 0 (leading coefficient is non-zero)
//   - remainder array has size max(M-1, 1) for the output
template <typename T>
void polynomial_divmod(
    T* quotient,
    T* remainder,
    const T* p,
    const T* q,
    int64_t N,
    int64_t M
) {
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const T leading_q = q[M - 1];

    // Initialize remainder with p
    // We use a working array that's the same size as p
    // After division, only the first rem_len coefficients are the actual remainder
    for (int64_t k = 0; k < N; ++k) {
        remainder[k] = p[k];
    }

    // Long division: work from highest degree to lowest
    for (int64_t i = quot_len - 1; i >= 0; --i) {
        // Position in remainder for this step
        const int64_t pos = M - 1 + i;  // = deg_q + i

        // Compute quotient coefficient
        const T q_coeff = remainder[pos] / leading_q;
        quotient[i] = q_coeff;

        // Subtract q_coeff * q * x^i from remainder
        // This affects remainder[i] through remainder[i + M - 1]
        for (int64_t j = 0; j < M; ++j) {
            remainder[i + j] -= q_coeff * q[j];
        }
    }

    // Note: The actual remainder is now in remainder[0..rem_len-1]
    // remainder[rem_len..N-1] should be zero (within numerical precision)
}

// Variant that takes a separate working buffer for remainder computation
// and outputs only the meaningful remainder coefficients
template <typename T>
void polynomial_divmod_with_buffer(
    T* quotient,
    T* remainder_out,  // size max(M-1, 1)
    T* work_buffer,    // size N (working space for division)
    const T* p,
    const T* q,
    int64_t N,
    int64_t M
) {
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const T leading_q = q[M - 1];

    // Initialize working buffer with p
    for (int64_t k = 0; k < N; ++k) {
        work_buffer[k] = p[k];
    }

    // Long division
    for (int64_t i = quot_len - 1; i >= 0; --i) {
        const int64_t pos = M - 1 + i;
        const T q_coeff = work_buffer[pos] / leading_q;
        quotient[i] = q_coeff;

        for (int64_t j = 0; j < M; ++j) {
            work_buffer[i + j] -= q_coeff * q[j];
        }
    }

    // Copy meaningful remainder
    for (int64_t k = 0; k < rem_len; ++k) {
        remainder_out[k] = work_buffer[k];
    }
}

// Complex specialization
template <typename T>
void polynomial_divmod(
    c10::complex<T>* quotient,
    c10::complex<T>* remainder,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M
) {
    const int64_t quot_len = N - M + 1;
    const c10::complex<T> leading_q = q[M - 1];

    for (int64_t k = 0; k < N; ++k) {
        remainder[k] = p[k];
    }

    for (int64_t i = quot_len - 1; i >= 0; --i) {
        const int64_t pos = M - 1 + i;
        const c10::complex<T> q_coeff = remainder[pos] / leading_q;
        quotient[i] = q_coeff;

        for (int64_t j = 0; j < M; ++j) {
            remainder[i + j] -= q_coeff * q[j];
        }
    }
}

template <typename T>
void polynomial_divmod_with_buffer(
    c10::complex<T>* quotient,
    c10::complex<T>* remainder_out,
    c10::complex<T>* work_buffer,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M
) {
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const c10::complex<T> leading_q = q[M - 1];

    for (int64_t k = 0; k < N; ++k) {
        work_buffer[k] = p[k];
    }

    for (int64_t i = quot_len - 1; i >= 0; --i) {
        const int64_t pos = M - 1 + i;
        const c10::complex<T> q_coeff = work_buffer[pos] / leading_q;
        quotient[i] = q_coeff;

        for (int64_t j = 0; j < M; ++j) {
            work_buffer[i + j] -= q_coeff * q[j];
        }
    }

    for (int64_t k = 0; k < rem_len; ++k) {
        remainder_out[k] = work_buffer[k];
    }
}

} // namespace torchscience::kernel::polynomial
