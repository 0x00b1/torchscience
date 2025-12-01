#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Legendre polynomial derivative
// The forward uses a recurrence that we need to reverse for gradients.
//
// Forward: output[j-1] = (2j-1)*c[j] after accumulation steps
// This is a linear operation, so we can compute gradients.
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coefficients, size N
//   grad_output: incoming gradient, size output_size
//   N: number of original coefficients
//   output_size: size of grad_output (should be N-1)
template <typename T>
void legendre_polynomial_p_derivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }

    // Initialize grad_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    if (N <= 1) {
        return;  // No gradient contribution
    }

    // The derivative operation is linear: d = L * c
    // So grad_c = L^T * grad_d
    //
    // From forward pass:
    // - der[0] = c[1]  ->  grad_c[1] += grad_d[0]
    // - der[1] = 3*c[2]  ->  grad_c[2] += 3*grad_d[1]
    // - For j >= 3: der[j-1] = (2j-1)*c[j] after accumulation
    //   The accumulation c[j-2] += c[j] means:
    //   grad_c[j] += grad from der[j-1] AND grad passed through to c[j-2]

    // Handle the basic terms (before accumulation effects)
    // grad_c[1] from der[0] = c[1]
    if (output_size > 0) {
        grad_coeffs[1] = grad_output[0];
    }

    // grad_c[2] from der[1] = 3*c[2]
    if (output_size > 1 && N > 2) {
        grad_coeffs[2] = T(3) * grad_output[1];
    }

    // For j >= 3: der[j-1] = (2j-1) * c_modified[j]
    // where c_modified[j] includes accumulation from higher terms
    // The backward through accumulation is complex - let's trace it forward.
    //
    // Actually, the forward modifies tmp during iteration.
    // Let's compute the effective linear transformation.
    //
    // The simplest approach: recompute the Jacobian action directly.
    // For Legendre derivative, the linear map is:
    // d/dc[k] (der[j]) depends on the accumulation pattern.
    //
    // A cleaner approach: note that for each c[j], its contribution flows:
    // - Directly to der[j-1] with factor (2j-1)
    // - Also to der[j-3], der[j-5], etc. through the accumulation
    //
    // The accumulation pattern for c[j]:
    // tmp[j] -> added to tmp[j-2] -> added to tmp[j-4] -> ...
    // Each tmp[k] with k >= 3 contributes (2k-1) to der[k-1]

    // Forward accumulation: for each j from N-1 down to 3:
    //   der[j-1] += (2j-1) * running_tmp[j]
    //   running_tmp[j-2] += running_tmp[j]
    //
    // Backward: we need to propagate gradients through this.
    // grad_running_tmp[j] = (2j-1) * grad_der[j-1]
    // grad_running_tmp[j] also gets += from grad_running_tmp[j-2]

    // Start with grad_tmp = 0
    T* grad_tmp = new T[N];
    for (int64_t k = 0; k < N; ++k) {
        grad_tmp[k] = T(0);
    }

    // Reverse the forward loop (j goes from 3 to N-1)
    for (int64_t j = 3; j <= N - 1; ++j) {
        // Forward: der[j-1] = (2j-1) * tmp[j]
        //          tmp[j-2] += tmp[j]
        // Backward:
        if (j - 1 < output_size) {
            grad_tmp[j] = grad_tmp[j] + T(2 * j - 1) * grad_output[j - 1];
        }
        // Also propagate from grad_tmp[j-2] (from next iteration's view)
    }

    // Actually the accumulation propagation is tricky.
    // Let me redo this more carefully.
    // Forward loop (j from N-1 down to 3):
    //   Step 1: der[j-1] = (2j-1) * tmp[j]
    //   Step 2: tmp[j-2] += tmp[j]
    //
    // After forward pass, tmp has been modified.
    // Backward (j from 3 up to N-1):
    //   Back-step 2: grad_tmp[j] += grad_tmp[j-2]
    //   Back-step 1: grad_tmp[j] += (2j-1) * grad_der[j-1]

    // Reset and redo properly
    for (int64_t k = 0; k < N; ++k) {
        grad_tmp[k] = T(0);
    }

    // Process in reverse order of forward (forward was N-1 down to 3, so backward is 3 up to N-1)
    for (int64_t j = 3; j <= N - 1; ++j) {
        // Backward for: tmp[j-2] += tmp[j]
        // grad_tmp[j] += grad_tmp[j-2] (already accumulated)
        // Then we need to backward for: der[j-1] = (2j-1) * tmp[j]
        if (j - 1 < output_size) {
            grad_tmp[j] = grad_tmp[j] + T(2 * j - 1) * grad_output[j - 1];
        }
        // Propagate gradient from j-2 to j (for the addition)
        grad_tmp[j] = grad_tmp[j] + grad_tmp[j - 2];
    }

    // The accumulation is still wrong. Let me trace through manually.
    // Forward: j=N-1: der[N-2] = (2N-3)*tmp[N-1]; tmp[N-3] += tmp[N-1]
    //          j=N-2: der[N-3] = (2N-5)*tmp[N-2]; tmp[N-4] += tmp[N-2]
    //          ...
    //          j=3: der[2] = 5*tmp[3]; tmp[1] += tmp[3]
    //
    // Then: der[1] = 3*tmp[2]; der[0] = tmp[1]
    //
    // grad flows: grad_der -> grad_tmp
    // grad_tmp[1] += grad_der[0]
    // grad_tmp[2] += 3*grad_der[1]
    // For j from 3 to N-1 (reverse of forward):
    //   grad_tmp[j] += (2j-1)*grad_der[j-1]
    //   grad_tmp[j] += grad_tmp[j-2]  (from the +=)

    // Reset again
    for (int64_t k = 0; k < N; ++k) {
        grad_tmp[k] = T(0);
    }

    // Start from the bottom
    if (output_size > 0 && N > 1) {
        grad_tmp[1] = grad_tmp[1] + grad_output[0];
    }
    if (output_size > 1 && N > 2) {
        grad_tmp[2] = grad_tmp[2] + T(3) * grad_output[1];
    }

    // Now process j from 3 to N-1
    for (int64_t j = 3; j <= N - 1; ++j) {
        // Gradient from der[j-1] = (2j-1) * tmp[j]
        if (j - 1 < output_size) {
            grad_tmp[j] = grad_tmp[j] + T(2 * j - 1) * grad_output[j - 1];
        }
        // Gradient from accumulation: tmp[j-2] += tmp[j]
        // This means d(tmp[j-2])/d(tmp[j]) = 1
        // So grad_tmp[j] += grad_tmp[j-2]
        grad_tmp[j] = grad_tmp[j] + grad_tmp[j - 2];
    }

    // grad_coeffs = grad_tmp (since tmp started as copy of coeffs)
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_tmp[k];
    }

    delete[] grad_tmp;
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_derivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }

    const c10::complex<T> zero(T(0), T(0));

    // Initialize grad_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    if (N <= 1) {
        return;
    }

    c10::complex<T>* grad_tmp = new c10::complex<T>[N];
    for (int64_t k = 0; k < N; ++k) {
        grad_tmp[k] = zero;
    }

    if (output_size > 0 && N > 1) {
        grad_tmp[1] = grad_tmp[1] + grad_output[0];
    }
    if (output_size > 1 && N > 2) {
        grad_tmp[2] = grad_tmp[2] + c10::complex<T>(T(3), T(0)) * grad_output[1];
    }

    for (int64_t j = 3; j <= N - 1; ++j) {
        if (j - 1 < output_size) {
            grad_tmp[j] = grad_tmp[j] + c10::complex<T>(T(2 * j - 1), T(0)) * grad_output[j - 1];
        }
        grad_tmp[j] = grad_tmp[j] + grad_tmp[j - 2];
    }

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_tmp[k];
    }

    delete[] grad_tmp;
}

} // namespace torchscience::kernel::polynomial
