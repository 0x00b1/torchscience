#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <torch/library.h>

namespace torchscience::cpu::filter {

namespace {

// ============================================================================
// Constants and Helper Structures
// ============================================================================

constexpr double kPi = 3.14159265358979323846;

/**
 * Structure to hold bandpass pole information for gradient computation.
 */
template <typename T>
struct BandpassPole {
    c10::complex<T> pole;      // The bandpass pole
    c10::complex<T> lp_pole;   // The lowpass pole it came from
    c10::complex<T> sqrt_disc; // sqrt((B*p/2)^2 - omega_0^2)
    bool is_bp1;               // true if pole = half_Bp + sqrt_disc, false if pole = half_Bp - sqrt_disc
};

// ============================================================================
// Forward Implementation
// ============================================================================

/**
 * Butterworth analog bandpass filter - compute SOS coefficients.
 *
 * This function computes the second-order sections for an analog Butterworth
 * bandpass filter given the filter order and passband edge frequencies.
 *
 * ALGORITHM:
 * 1. Generate n lowpass prototype poles on the unit circle
 * 2. For each conjugate pair of lowpass poles, transform to get 4 bandpass poles
 * 3. These 4 poles form 2 conjugate pairs -> 2 SOS sections
 * 4. Normalize for unity gain at center frequency
 *
 * @param n Filter order (positive integer)
 * @param omega_p1 Lower passband frequency
 * @param omega_p2 Upper passband frequency
 * @param sos Output array of SOS coefficients, shape (n, 6)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter_kernel(
    int64_t n,
    T omega_p1,
    T omega_p2,
    T* sos  // Output: n * 6 elements
) {
    const T pi = T(kPi);

    // Compute center frequency and bandwidth
    T omega_0 = std::sqrt(omega_p1 * omega_p2);
    T omega_0_sq = omega_p1 * omega_p2;
    T B = omega_p2 - omega_p1;

    // Clamp n to max order
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Process lowpass poles to generate SOS sections
    // For each pair of conjugate lowpass poles, we get 2 SOS sections
    int64_t section_idx = 0;

    // Number of lowpass pole pairs to process
    // For even n: n/2 pairs, each giving 2 sections -> n sections total
    // For odd n: (n-1)/2 pairs + 1 middle pole -> (n-1) + 1 = n sections
    int64_t num_pairs = actual_n / 2;
    bool has_middle = (actual_n % 2 == 1);

    // Process conjugate pairs of lowpass poles
    for (int64_t k = 0; k < num_pairs; ++k) {
        // Lowpass pole index (1-based in the original formula)
        int64_t k1 = k + 1;

        // Compute lowpass pole angle
        T angle = pi * (T(2 * k1) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        // Compute bandpass poles
        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // Section from bp1 and conj(bp1)
        // a1 = -2 * Re(bp1), a2 = |bp1|^2
        T a1_bp1 = -T(2) * bp1.real();
        T a2_bp1 = bp1.real() * bp1.real() + bp1.imag() * bp1.imag();

        T* sos_row = sos + section_idx * 6;
        sos_row[0] = T(1);  // b0 (will be normalized later)
        sos_row[1] = T(0);  // b1
        sos_row[2] = T(0);  // b2
        sos_row[3] = T(1);  // a0
        sos_row[4] = a1_bp1;
        sos_row[5] = a2_bp1;
        section_idx++;

        // Section from bp2 and conj(bp2)
        T a1_bp2 = -T(2) * bp2.real();
        T a2_bp2 = bp2.real() * bp2.real() + bp2.imag() * bp2.imag();

        sos_row = sos + section_idx * 6;
        sos_row[0] = T(1);  // b0 (will be normalized later)
        sos_row[1] = T(0);  // b1
        sos_row[2] = T(0);  // b2
        sos_row[3] = T(1);  // a0
        sos_row[4] = a1_bp2;
        sos_row[5] = a2_bp2;
        section_idx++;
    }

    // Handle middle pole for odd n
    if (has_middle) {
        // Middle lowpass pole is at angle = pi (real, = -1)
        int64_t k_mid = num_pairs + 1;
        T angle = pi * (T(2 * k_mid) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        // For real lowpass pole, the two bandpass poles are either
        // both real or complex conjugates
        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // These should be conjugates of each other (or both real)
        // Either way, they form one SOS section
        // a1 = -(bp1 + bp2) = -2*half_Bp (which is real since lp_pole is real)
        // a2 = bp1 * bp2 = half_Bp^2 - disc = omega_0^2
        T a1 = -(bp1 + bp2).real();
        T a2 = (bp1 * bp2).real();

        T* sos_row = sos + section_idx * 6;
        sos_row[0] = T(1);  // b0
        sos_row[1] = T(0);  // b1
        sos_row[2] = T(0);  // b2
        sos_row[3] = T(1);  // a0
        sos_row[4] = a1;
        sos_row[5] = a2;
        section_idx++;
    }

    // Compute total gain at center frequency and normalize
    // At s = j*omega_0, for section with numerator s^2 and denominator s^2 + a1*s + a2:
    // H(j*omega_0) = (j*omega_0)^2 / ((j*omega_0)^2 + a1*(j*omega_0) + a2)
    //              = -omega_0^2 / (-omega_0^2 + j*a1*omega_0 + a2)
    //              = -omega_0^2 / ((a2 - omega_0^2) + j*a1*omega_0)
    T total_gain_sq = T(1);
    for (int64_t i = 0; i < actual_n; ++i) {
        T* sos_row = sos + i * 6;
        T a1 = sos_row[4];
        T a2 = sos_row[5];

        // |H(j*omega_0)|^2 = omega_0^4 / ((a2 - omega_0^2)^2 + a1^2*omega_0^2)
        T num_sq = omega_0_sq * omega_0_sq;
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        total_gain_sq *= num_sq / den_sq;
    }
    T total_gain = std::sqrt(total_gain_sq);

    // Distribute normalization factor across all sections
    T normalization_factor = T(1) / std::pow(total_gain, T(1) / T(actual_n));
    for (int64_t i = 0; i < actual_n; ++i) {
        T* sos_row = sos + i * 6;
        sos_row[0] *= normalization_factor;  // b0
    }
}

// ============================================================================
// Backward Implementation (First-Order Derivatives)
// ============================================================================

/**
 * Backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes analytical gradients w.r.t. omega_p1 and omega_p2 given the
 * gradient w.r.t. the SOS output.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter_backward_kernel(
    const T* grad_sos,  // Input: n * 6 elements
    int64_t n,
    T omega_p1,
    T omega_p2,
    T& grad_omega_p1,
    T& grad_omega_p2
) {
    const T pi = T(kPi);
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Compute intermediate quantities
    T omega_0_sq = omega_p1 * omega_p2;
    T omega_0 = std::sqrt(omega_0_sq);
    T B = omega_p2 - omega_p1;

    // We need to recompute the forward pass to get the poles and normalization
    // This is necessary for correct gradient computation

    // First, compute all poles and their derivatives
    struct PoleInfo {
        c10::complex<T> bp;       // bandpass pole
        c10::complex<T> lp;       // lowpass pole
        c10::complex<T> sqrt_disc;// sqrt of discriminant
        bool is_bp1;              // true for bp1, false for bp2
    };

    PoleInfo poles[kMaxOrder];
    T a1_vals[kMaxOrder];
    T a2_vals[kMaxOrder];

    int64_t num_pairs = actual_n / 2;
    bool has_middle = (actual_n % 2 == 1);
    int64_t section_idx = 0;

    // Compute poles for each section
    for (int64_t k = 0; k < num_pairs; ++k) {
        int64_t k1 = k + 1;
        T angle = pi * (T(2 * k1) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // Section from bp1
        poles[section_idx] = {bp1, lp_pole, sqrt_disc, true};
        a1_vals[section_idx] = -T(2) * bp1.real();
        a2_vals[section_idx] = bp1.real() * bp1.real() + bp1.imag() * bp1.imag();
        section_idx++;

        // Section from bp2
        poles[section_idx] = {bp2, lp_pole, sqrt_disc, false};
        a1_vals[section_idx] = -T(2) * bp2.real();
        a2_vals[section_idx] = bp2.real() * bp2.real() + bp2.imag() * bp2.imag();
        section_idx++;
    }

    if (has_middle) {
        int64_t k_mid = num_pairs + 1;
        T angle = pi * (T(2 * k_mid) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // For middle pole, we use bp1 and bp2 together in one section
        // a1 = -(bp1 + bp2) = -2*half_Bp
        // a2 = bp1 * bp2 = omega_0^2
        poles[section_idx] = {bp1, lp_pole, sqrt_disc, true};  // Store bp1, derivative is special
        a1_vals[section_idx] = -(bp1 + bp2).real();
        a2_vals[section_idx] = (bp1 * bp2).real();
        section_idx++;
    }

    // Compute gain and normalization factor
    T total_gain_sq = T(1);
    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T num_sq = omega_0_sq * omega_0_sq;
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        total_gain_sq *= num_sq / den_sq;
    }
    T total_gain = std::sqrt(total_gain_sq);
    T nf = T(1) / std::pow(total_gain, T(1) / T(actual_n));

    // Now compute gradients
    grad_omega_p1 = T(0);
    grad_omega_p2 = T(0);

    // Gradient from a1 and a2 coefficients (before normalization effects)
    for (int64_t i = 0; i < actual_n; ++i) {
        const T* grad_row = grad_sos + i * 6;
        T grad_b0 = grad_row[0];
        T grad_a1 = grad_row[4];
        T grad_a2 = grad_row[5];

        c10::complex<T> lp = poles[i].lp;
        c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
        c10::complex<T> bp = poles[i].bp;
        bool is_bp1 = poles[i].is_bp1;

        // Handle middle pole specially
        bool is_middle = has_middle && (i == actual_n - 1);

        if (is_middle) {
            // For middle pole: a1 = -B*lp.real(), a2 = omega_0^2
            // Since lp is real (cos(pi) = -1), lp.real() = lp
            T lp_real = lp.real();

            // d(a1)/d(omega_p1) = -d(B)/d(omega_p1) * lp_real = lp_real
            // d(a1)/d(omega_p2) = -d(B)/d(omega_p2) * lp_real = -lp_real
            grad_omega_p1 += grad_a1 * lp_real;
            grad_omega_p2 += grad_a1 * (-lp_real);

            // d(a2)/d(omega_p1) = omega_p2
            // d(a2)/d(omega_p2) = omega_p1
            grad_omega_p1 += grad_a2 * omega_p2;
            grad_omega_p2 += grad_a2 * omega_p1;
        } else {
            // For conjugate-paired poles:
            // bp = (B/2)*lp + sign * sqrt((B*lp/2)^2 - omega_0^2)
            // where sign = +1 for bp1, -1 for bp2

            c10::complex<T> half_Bp = (B * lp) / T(2);

            // d(half_Bp)/d(omega_p1) = -lp/2
            // d(half_Bp)/d(omega_p2) = lp/2
            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);

            // d(D)/d(omega_p1) = -half_Bp*lp - omega_p2
            // d(D)/d(omega_p2) = half_Bp*lp - omega_p1
            c10::complex<T> d_D_d_w1 = -half_Bp * lp - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = half_Bp * lp - c10::complex<T>(omega_p1, T(0));

            // d(sqrt_D)/d(omega) = d(D)/d(omega) / (2 * sqrt_D)
            c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
            c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
            c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

            // d(bp)/d(omega) depends on whether it's bp1 or bp2
            c10::complex<T> d_bp_d_w1, d_bp_d_w2;
            if (is_bp1) {
                d_bp_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;
            } else {
                d_bp_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;
            }

            // d(a1)/d(omega) = -2 * Re(d(bp)/d(omega))
            T d_a1_d_w1 = -T(2) * d_bp_d_w1.real();
            T d_a1_d_w2 = -T(2) * d_bp_d_w2.real();

            // d(a2)/d(omega) = 2 * Re(bp * conj(d(bp)/d(omega)))
            // Re(bp * conj(d_bp)) = Re(bp)*Re(d_bp) + Im(bp)*Im(d_bp)
            T d_a2_d_w1 = T(2) * (bp.real() * d_bp_d_w1.real() + bp.imag() * d_bp_d_w1.imag());
            T d_a2_d_w2 = T(2) * (bp.real() * d_bp_d_w2.real() + bp.imag() * d_bp_d_w2.imag());

            grad_omega_p1 += grad_a1 * d_a1_d_w1 + grad_a2 * d_a2_d_w1;
            grad_omega_p2 += grad_a1 * d_a1_d_w2 + grad_a2 * d_a2_d_w2;
        }

        // Gradient from b0 (normalization factor)
        // b0 = nf for all sections
        // nf = 1 / total_gain^(1/n)
        // total_gain = sqrt(prod_i |H_i(j*omega_0)|^2)
        // |H_i|^2 = omega_0^4 / ((a2_i - omega_0^2)^2 + a1_i^2 * omega_0^2)
        //
        // This requires computing d(nf)/d(omega) which involves the chain rule
        // through all the gain terms. For simplicity, we compute this separately.
    }

    // Compute gradient of normalization factor
    // nf = total_gain^(-1/n)
    // d(nf)/d(omega) = (-1/n) * total_gain^(-1/n - 1) * d(total_gain)/d(omega)
    //                = (-1/n) * nf / total_gain * d(total_gain)/d(omega)
    //                = (-nf / (n * total_gain)) * d(total_gain)/d(omega)
    //
    // total_gain = sqrt(total_gain_sq)
    // d(total_gain)/d(omega) = d(total_gain_sq)/d(omega) / (2 * total_gain)
    //
    // total_gain_sq = prod_i (omega_0^4 / den_sq_i)
    // log(total_gain_sq) = sum_i (4*log(omega_0) - log(den_sq_i))
    // d(log(total_gain_sq))/d(omega) = 4*n*d(log(omega_0))/d(omega) - sum_i d(log(den_sq_i))/d(omega)
    //                                = 2*n/omega_0 * d(omega_0)/d(omega) - sum_i (1/den_sq_i)*d(den_sq_i)/d(omega)

    // d(omega_0)/d(omega_p1) = omega_p2 / (2*omega_0)
    // d(omega_0)/d(omega_p2) = omega_p1 / (2*omega_0)
    T d_omega0_d_w1 = omega_p2 / (T(2) * omega_0);
    T d_omega0_d_w2 = omega_p1 / (T(2) * omega_0);

    // Compute d(log(total_gain_sq))/d(omega)
    // log(total_gain_sq) = sum_i (4*log(omega_0) - log(den_sq_i))
    //                    = 4*n*log(omega_0) - sum_i log(den_sq_i)
    // d(log(total_gain_sq))/d(omega) = 4*n/omega_0 * d(omega_0)/d(omega) - sum_i (1/den_sq_i) * d(den_sq_i)/d(omega)
    T d_log_gain_sq_d_w1 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w1;
    T d_log_gain_sq_d_w2 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w2;

    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;

        // d(den_sq)/d(omega) = 2*den_real*d(den_real)/d(omega) + 2*a1*omega_0_sq*d(a1)/d(omega) + a1^2*2*omega_0*d(omega_0)/d(omega)
        // d(den_real)/d(omega) = d(a2)/d(omega) - 2*omega_0*d(omega_0)/d(omega)

        // We need d(a1)/d(omega) and d(a2)/d(omega) for this section
        // These are the same as computed above, but we need to store or recompute them

        c10::complex<T> lp = poles[i].lp;
        c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
        c10::complex<T> bp = poles[i].bp;
        bool is_bp1 = poles[i].is_bp1;
        bool is_middle = has_middle && (i == actual_n - 1);

        T d_a1_d_w1, d_a1_d_w2, d_a2_d_w1, d_a2_d_w2;

        if (is_middle) {
            T lp_real = lp.real();
            d_a1_d_w1 = lp_real;
            d_a1_d_w2 = -lp_real;
            d_a2_d_w1 = omega_p2;
            d_a2_d_w2 = omega_p1;
        } else {
            c10::complex<T> half_Bp = (B * lp) / T(2);
            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);
            c10::complex<T> d_D_d_w1 = -half_Bp * lp - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = half_Bp * lp - c10::complex<T>(omega_p1, T(0));
            c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
            c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
            c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

            c10::complex<T> d_bp_d_w1, d_bp_d_w2;
            if (is_bp1) {
                d_bp_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;
            } else {
                d_bp_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;
            }

            d_a1_d_w1 = -T(2) * d_bp_d_w1.real();
            d_a1_d_w2 = -T(2) * d_bp_d_w2.real();
            d_a2_d_w1 = T(2) * (bp.real() * d_bp_d_w1.real() + bp.imag() * d_bp_d_w1.imag());
            d_a2_d_w2 = T(2) * (bp.real() * d_bp_d_w2.real() + bp.imag() * d_bp_d_w2.imag());
        }

        // d(den_real)/d(omega) = d(a2)/d(omega) - 2*omega_0*d(omega_0)/d(omega)
        T d_den_real_d_w1 = d_a2_d_w1 - T(2) * omega_0 * d_omega0_d_w1;
        T d_den_real_d_w2 = d_a2_d_w2 - T(2) * omega_0 * d_omega0_d_w2;

        // d(den_sq)/d(omega) = 2*den_real*d(den_real)/d(omega) + 2*a1*omega_0_sq*d(a1)/d(omega) + 2*a1^2*omega_0*d(omega_0)/d(omega)
        T d_den_sq_d_w1 = T(2) * den_real * d_den_real_d_w1
                       + T(2) * a1 * omega_0_sq * d_a1_d_w1
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w1;
        T d_den_sq_d_w2 = T(2) * den_real * d_den_real_d_w2
                       + T(2) * a1 * omega_0_sq * d_a1_d_w2
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w2;

        // Subtract contribution to d(log(total_gain_sq))
        d_log_gain_sq_d_w1 -= d_den_sq_d_w1 / den_sq;
        d_log_gain_sq_d_w2 -= d_den_sq_d_w2 / den_sq;
    }

    // d(total_gain_sq)/d(omega) = total_gain_sq * d(log(total_gain_sq))/d(omega)
    // d(total_gain)/d(omega) = d(total_gain_sq)/d(omega) / (2*total_gain)
    //                        = total_gain_sq * d(log(total_gain_sq))/d(omega) / (2*total_gain)
    //                        = total_gain * d(log(total_gain_sq))/d(omega) / 2
    T d_total_gain_d_w1 = total_gain * d_log_gain_sq_d_w1 / T(2);
    T d_total_gain_d_w2 = total_gain * d_log_gain_sq_d_w2 / T(2);

    // d(nf)/d(omega) = (-nf / (n * total_gain)) * d(total_gain)/d(omega)
    T d_nf_d_w1 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w1;
    T d_nf_d_w2 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w2;

    // Add contribution from grad_b0 for all sections
    T total_grad_b0 = T(0);
    for (int64_t i = 0; i < actual_n; ++i) {
        total_grad_b0 += grad_sos[i * 6];
    }

    grad_omega_p1 += total_grad_b0 * d_nf_d_w1;
    grad_omega_p2 += total_grad_b0 * d_nf_d_w2;
}

// ============================================================================
// Double-Backward Implementation (Second-Order Derivatives)
// ============================================================================

/**
 * Double-backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes second-order gradients for Hessian-vector products.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter_backward_backward_kernel(
    T grad_grad_omega_p1,
    T grad_grad_omega_p2,
    const T* grad_sos,
    int64_t n,
    T omega_p1,
    T omega_p2,
    bool has_grad_grad_omega_p1,
    bool has_grad_grad_omega_p2,
    T* grad_grad_sos,  // Output: n * 6 elements
    T& new_grad_omega_p1,
    T& new_grad_omega_p2
) {
    const T pi = T(kPi);
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Initialize outputs
    new_grad_omega_p1 = T(0);
    new_grad_omega_p2 = T(0);
    for (int64_t i = 0; i < actual_n * 6; ++i) {
        grad_grad_sos[i] = T(0);
    }

    // Early exit if no gradients
    if (!has_grad_grad_omega_p1 && !has_grad_grad_omega_p2) {
        return;
    }

    // Compute intermediate quantities
    T omega_0_sq = omega_p1 * omega_p2;
    T omega_0 = std::sqrt(omega_0_sq);
    T B = omega_p2 - omega_p1;

    // Derivatives of omega_0 and omega_0_sq
    T d_omega0_d_w1 = omega_p2 / (T(2) * omega_0);
    T d_omega0_d_w2 = omega_p1 / (T(2) * omega_0);

    // Second derivatives of omega_0
    T omega_0_cubed = omega_0 * omega_0_sq;
    T d2_omega0_d_w1_w1 = -omega_p2 * omega_p2 / (T(4) * omega_0_cubed);
    T d2_omega0_d_w2_w2 = -omega_p1 * omega_p1 / (T(4) * omega_0_cubed);
    T d2_omega0_d_w1_w2 = T(1) / (T(4) * omega_0);

    // Recompute pole information
    struct PoleInfo {
        c10::complex<T> bp;
        c10::complex<T> lp;
        c10::complex<T> sqrt_disc;
        c10::complex<T> half_Bp;
        bool is_bp1;
    };

    PoleInfo poles[kMaxOrder];
    T a1_vals[kMaxOrder];
    T a2_vals[kMaxOrder];

    // First derivatives storage
    T d_a1_d_w1[kMaxOrder], d_a1_d_w2[kMaxOrder];
    T d_a2_d_w1[kMaxOrder], d_a2_d_w2[kMaxOrder];

    int64_t num_pairs = actual_n / 2;
    bool has_middle = (actual_n % 2 == 1);
    int64_t section_idx = 0;

    // Compute poles and first derivatives for each section
    for (int64_t k = 0; k < num_pairs; ++k) {
        int64_t k1 = k + 1;
        T angle = pi * (T(2 * k1) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // First derivatives of half_Bp
        c10::complex<T> d_half_Bp_d_w1 = -lp_pole / T(2);
        c10::complex<T> d_half_Bp_d_w2 = lp_pole / T(2);

        // First derivatives of disc D = half_Bp² - omega_0²
        c10::complex<T> d_D_d_w1 = T(2) * half_Bp * d_half_Bp_d_w1 - c10::complex<T>(omega_p2, T(0));
        c10::complex<T> d_D_d_w2 = T(2) * half_Bp * d_half_Bp_d_w2 - c10::complex<T>(omega_p1, T(0));

        // First derivatives of sqrt_disc
        c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
        c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
        c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

        // Section from bp1
        poles[section_idx] = {bp1, lp_pole, sqrt_disc, half_Bp, true};
        a1_vals[section_idx] = -T(2) * bp1.real();
        a2_vals[section_idx] = bp1.real() * bp1.real() + bp1.imag() * bp1.imag();

        c10::complex<T> d_bp1_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
        c10::complex<T> d_bp1_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;

        d_a1_d_w1[section_idx] = -T(2) * d_bp1_d_w1.real();
        d_a1_d_w2[section_idx] = -T(2) * d_bp1_d_w2.real();
        d_a2_d_w1[section_idx] = T(2) * (bp1.real() * d_bp1_d_w1.real() + bp1.imag() * d_bp1_d_w1.imag());
        d_a2_d_w2[section_idx] = T(2) * (bp1.real() * d_bp1_d_w2.real() + bp1.imag() * d_bp1_d_w2.imag());
        section_idx++;

        // Section from bp2
        poles[section_idx] = {bp2, lp_pole, sqrt_disc, half_Bp, false};
        a1_vals[section_idx] = -T(2) * bp2.real();
        a2_vals[section_idx] = bp2.real() * bp2.real() + bp2.imag() * bp2.imag();

        c10::complex<T> d_bp2_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
        c10::complex<T> d_bp2_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;

        d_a1_d_w1[section_idx] = -T(2) * d_bp2_d_w1.real();
        d_a1_d_w2[section_idx] = -T(2) * d_bp2_d_w2.real();
        d_a2_d_w1[section_idx] = T(2) * (bp2.real() * d_bp2_d_w1.real() + bp2.imag() * d_bp2_d_w1.imag());
        d_a2_d_w2[section_idx] = T(2) * (bp2.real() * d_bp2_d_w2.real() + bp2.imag() * d_bp2_d_w2.imag());
        section_idx++;
    }

    if (has_middle) {
        int64_t k_mid = num_pairs + 1;
        T angle = pi * (T(2 * k_mid) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));
        T lp_real = lp_pole.real();

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);
        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        poles[section_idx] = {bp1, lp_pole, sqrt_disc, half_Bp, true};
        a1_vals[section_idx] = -(bp1 + bp2).real();  // = -B * lp_real
        a2_vals[section_idx] = (bp1 * bp2).real();   // = omega_0²

        // For middle pole: a1 = -B*lp_real, a2 = omega_0²
        d_a1_d_w1[section_idx] = lp_real;   // d(-B*lp)/dw1 = lp
        d_a1_d_w2[section_idx] = -lp_real;  // d(-B*lp)/dw2 = -lp
        d_a2_d_w1[section_idx] = omega_p2;  // d(omega_0²)/dw1 = w2
        d_a2_d_w2[section_idx] = omega_p1;  // d(omega_0²)/dw2 = w1
        section_idx++;
    }

    // Compute gain and normalization factor
    T total_gain_sq = T(1);
    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T num_sq = omega_0_sq * omega_0_sq;
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        total_gain_sq *= num_sq / den_sq;
    }
    T total_gain = std::sqrt(total_gain_sq);
    T nf = T(1) / std::pow(total_gain, T(1) / T(actual_n));

    // Compute first derivatives of normalization factor (d_nf_d_w1, d_nf_d_w2)
    T d_log_gain_sq_d_w1 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w1;
    T d_log_gain_sq_d_w2 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w2;

    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;

        T d_den_real_d_w1 = d_a2_d_w1[i] - T(2) * omega_0 * d_omega0_d_w1;
        T d_den_real_d_w2 = d_a2_d_w2[i] - T(2) * omega_0 * d_omega0_d_w2;

        T d_den_sq_d_w1 = T(2) * den_real * d_den_real_d_w1
                       + T(2) * a1 * omega_0_sq * d_a1_d_w1[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w1;
        T d_den_sq_d_w2 = T(2) * den_real * d_den_real_d_w2
                       + T(2) * a1 * omega_0_sq * d_a1_d_w2[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w2;

        d_log_gain_sq_d_w1 -= d_den_sq_d_w1 / den_sq;
        d_log_gain_sq_d_w2 -= d_den_sq_d_w2 / den_sq;
    }

    T d_total_gain_d_w1 = total_gain * d_log_gain_sq_d_w1 / T(2);
    T d_total_gain_d_w2 = total_gain * d_log_gain_sq_d_w2 / T(2);

    T d_nf_d_w1 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w1;
    T d_nf_d_w2 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w2;

    // =========================================================================
    // Compute grad_grad_sos: contributions from Jacobian terms
    // grad_grad_sos[i] = grad_grad_omega_p1 * J_1i + grad_grad_omega_p2 * J_2i
    // =========================================================================
    for (int64_t i = 0; i < actual_n; ++i) {
        T* gg_row = grad_grad_sos + i * 6;

        // b0 coefficient gets gradient from normalization factor
        gg_row[0] = (has_grad_grad_omega_p1 ? grad_grad_omega_p1 * d_nf_d_w1 : T(0))
                  + (has_grad_grad_omega_p2 ? grad_grad_omega_p2 * d_nf_d_w2 : T(0));

        // b1, b2, a0 are constants (0, 0, 1) - no gradient
        gg_row[1] = T(0);
        gg_row[2] = T(0);
        gg_row[3] = T(0);

        // a1, a2 get gradients from pole derivatives
        gg_row[4] = (has_grad_grad_omega_p1 ? grad_grad_omega_p1 * d_a1_d_w1[i] : T(0))
                  + (has_grad_grad_omega_p2 ? grad_grad_omega_p2 * d_a1_d_w2[i] : T(0));
        gg_row[5] = (has_grad_grad_omega_p1 ? grad_grad_omega_p1 * d_a2_d_w1[i] : T(0))
                  + (has_grad_grad_omega_p2 ? grad_grad_omega_p2 * d_a2_d_w2[i] : T(0));
    }

    // =========================================================================
    // Compute new_grad_omega_p1, new_grad_omega_p2 from Hessian terms
    // These require second derivatives of a1, a2, and nf
    // =========================================================================
    for (int64_t i = 0; i < actual_n; ++i) {
        const T* grad_row = grad_sos + i * 6;
        T grad_a1 = grad_row[4];
        T grad_a2 = grad_row[5];

        c10::complex<T> lp = poles[i].lp;
        c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
        c10::complex<T> half_Bp = poles[i].half_Bp;
        c10::complex<T> bp = poles[i].bp;
        bool is_bp1 = poles[i].is_bp1;
        bool is_middle = has_middle && (i == actual_n - 1);

        T d2_a1_d_w1_w1, d2_a1_d_w2_w2, d2_a1_d_w1_w2;
        T d2_a2_d_w1_w1, d2_a2_d_w2_w2, d2_a2_d_w1_w2;

        if (is_middle) {
            // For middle pole: a1 = -B*lp_real (linear in B), a2 = omega_0²
            // Second derivatives of a1 are all zero
            d2_a1_d_w1_w1 = T(0);
            d2_a1_d_w2_w2 = T(0);
            d2_a1_d_w1_w2 = T(0);

            // Second derivatives of a2 = omega_0² = w1*w2
            d2_a2_d_w1_w1 = T(0);
            d2_a2_d_w2_w2 = T(0);
            d2_a2_d_w1_w2 = T(1);
        } else {
            // Compute second derivatives of bandpass poles
            // D = half_Bp² - omega_0²
            // sqrt_D = sqrt(D)
            // bp = half_Bp ± sqrt_D

            // Second derivatives of D
            c10::complex<T> lp_sq = lp * lp;
            c10::complex<T> d2_D_d_w1_w1 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w2_w2 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w1_w2 = -lp_sq / T(2) - c10::complex<T>(T(1), T(0));

            // First derivatives of D (recompute for clarity)
            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);
            c10::complex<T> d_D_d_w1 = T(2) * half_Bp * d_half_Bp_d_w1 - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = T(2) * half_Bp * d_half_Bp_d_w2 - c10::complex<T>(omega_p1, T(0));

            c10::complex<T> D = sqrt_disc * sqrt_disc;

            // Second derivatives of sqrt_D using quotient rule:
            // d(sqrt_D)/dw = dD/dw / (2*sqrt_D)
            // d²(sqrt_D)/dw² = [d²D/dw² * 2*sqrt_D - dD/dw * dD/dw / sqrt_D] / (4*D)
            //                = [2*D * d²D/dw² - (dD/dw)²] / (4 * D^(3/2))

            c10::complex<T> D_cubed_half = D * sqrt_disc;  // D^(3/2)
            c10::complex<T> four_D_cubed_half = T(4) * D_cubed_half;

            // Avoid division by zero for small D
            T D_mag = std::abs(D);
            if (D_mag < std::numeric_limits<T>::epsilon() * T(100)) {
                // Near-zero discriminant: second derivatives are ill-defined
                // Use zero as a safe approximation
                d2_a1_d_w1_w1 = T(0);
                d2_a1_d_w2_w2 = T(0);
                d2_a1_d_w1_w2 = T(0);
                d2_a2_d_w1_w1 = T(0);
                d2_a2_d_w2_w2 = T(0);
                d2_a2_d_w1_w2 = T(0);
            } else {
                c10::complex<T> d2_sqrt_d_w1_w1 = (T(2) * D * d2_D_d_w1_w1 - d_D_d_w1 * d_D_d_w1) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w2_w2 = (T(2) * D * d2_D_d_w2_w2 - d_D_d_w2 * d_D_d_w2) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w1_w2 = (T(2) * D * d2_D_d_w1_w2 - d_D_d_w1 * d_D_d_w2) / four_D_cubed_half;

                // Second derivatives of bp = half_Bp ± sqrt_D
                // d²(half_Bp)/dw² = 0 for all combinations
                c10::complex<T> d2_bp_d_w1_w1, d2_bp_d_w2_w2, d2_bp_d_w1_w2;
                if (is_bp1) {
                    d2_bp_d_w1_w1 = d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = d2_sqrt_d_w1_w2;
                } else {
                    d2_bp_d_w1_w1 = -d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = -d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = -d2_sqrt_d_w1_w2;
                }

                // First derivatives of bp (recompute)
                c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
                c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
                c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

                c10::complex<T> d_bp_d_w1, d_bp_d_w2;
                if (is_bp1) {
                    d_bp_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
                    d_bp_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;
                } else {
                    d_bp_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
                    d_bp_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;
                }

                // Second derivatives of a1 = -2 * Re(bp)
                d2_a1_d_w1_w1 = -T(2) * d2_bp_d_w1_w1.real();
                d2_a1_d_w2_w2 = -T(2) * d2_bp_d_w2_w2.real();
                d2_a1_d_w1_w2 = -T(2) * d2_bp_d_w1_w2.real();

                // Second derivatives of a2 = |bp|² = Re(bp)² + Im(bp)²
                // d(a2)/dw = 2 * Re(bp * conj(d_bp/dw)) = 2*(Re(bp)*Re(d_bp) + Im(bp)*Im(d_bp))
                // d²(a2)/dw² = 2 * [Re(d_bp/dw)² + Im(d_bp/dw)² + Re(bp)*Re(d²_bp/dw²) + Im(bp)*Im(d²_bp/dw²)]
                //            = 2 * [|d_bp/dw|² + Re(bp * conj(d²_bp/dw²))]

                T d2_a2_contrib_11 = d_bp_d_w1.real() * d_bp_d_w1.real() + d_bp_d_w1.imag() * d_bp_d_w1.imag()
                                   + bp.real() * d2_bp_d_w1_w1.real() + bp.imag() * d2_bp_d_w1_w1.imag();
                T d2_a2_contrib_22 = d_bp_d_w2.real() * d_bp_d_w2.real() + d_bp_d_w2.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w2_w2.real() + bp.imag() * d2_bp_d_w2_w2.imag();
                T d2_a2_contrib_12 = d_bp_d_w1.real() * d_bp_d_w2.real() + d_bp_d_w1.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w1_w2.real() + bp.imag() * d2_bp_d_w1_w2.imag();

                d2_a2_d_w1_w1 = T(2) * d2_a2_contrib_11;
                d2_a2_d_w2_w2 = T(2) * d2_a2_contrib_22;
                d2_a2_d_w1_w2 = T(2) * d2_a2_contrib_12;
            }
        }

        // Accumulate Hessian contributions
        if (has_grad_grad_omega_p1) {
            new_grad_omega_p1 += grad_a1 * d2_a1_d_w1_w1 * grad_grad_omega_p1;
            new_grad_omega_p1 += grad_a2 * d2_a2_d_w1_w1 * grad_grad_omega_p1;
            new_grad_omega_p2 += grad_a1 * d2_a1_d_w1_w2 * grad_grad_omega_p1;
            new_grad_omega_p2 += grad_a2 * d2_a2_d_w1_w2 * grad_grad_omega_p1;
        }
        if (has_grad_grad_omega_p2) {
            new_grad_omega_p1 += grad_a1 * d2_a1_d_w1_w2 * grad_grad_omega_p2;
            new_grad_omega_p1 += grad_a2 * d2_a2_d_w1_w2 * grad_grad_omega_p2;
            new_grad_omega_p2 += grad_a1 * d2_a1_d_w2_w2 * grad_grad_omega_p2;
            new_grad_omega_p2 += grad_a2 * d2_a2_d_w2_w2 * grad_grad_omega_p2;
        }
    }

    // Add Hessian contributions from normalization factor
    // nf = total_gain^(-1/n), where total_gain = sqrt(prod_i gain_i)
    // log(nf) = -1/(2n) * log(total_gain_sq)
    // log(total_gain_sq) = 4n*log(omega_0) - sum_i log(den_sq_i)

    T total_grad_b0 = T(0);
    for (int64_t i = 0; i < actual_n; ++i) {
        total_grad_b0 += grad_sos[i * 6];
    }

    // Second derivatives of log(omega_0)
    T d2_log_w0_d_w1_w1 = -d_omega0_d_w1 * d_omega0_d_w1 / omega_0_sq + d2_omega0_d_w1_w1 / omega_0;
    T d2_log_w0_d_w2_w2 = -d_omega0_d_w2 * d_omega0_d_w2 / omega_0_sq + d2_omega0_d_w2_w2 / omega_0;
    T d2_log_w0_d_w1_w2 = -d_omega0_d_w1 * d_omega0_d_w2 / omega_0_sq + d2_omega0_d_w1_w2 / omega_0;

    // Start with contribution from 4n*log(omega_0)
    T d2_log_gain_sq_d_w1_w1 = T(4) * T(actual_n) * d2_log_w0_d_w1_w1;
    T d2_log_gain_sq_d_w2_w2 = T(4) * T(actual_n) * d2_log_w0_d_w2_w2;
    T d2_log_gain_sq_d_w1_w2 = T(4) * T(actual_n) * d2_log_w0_d_w1_w2;

    // Subtract contributions from each log(den_sq_i)
    // For each section: den_sq = (a2 - omega_0²)² + a1² * omega_0²
    // d²log(den_sq)/dω² = d²(den_sq)/dω² / den_sq - (d(den_sq)/dω)² / den_sq²
    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        T den_sq_sq = den_sq * den_sq;

        // First derivatives of den_sq (already computed but recompute for clarity)
        T d_den_real_d_w1 = d_a2_d_w1[i] - T(2) * omega_0 * d_omega0_d_w1;
        T d_den_real_d_w2 = d_a2_d_w2[i] - T(2) * omega_0 * d_omega0_d_w2;

        T d_den_sq_d_w1 = T(2) * den_real * d_den_real_d_w1
                       + T(2) * a1 * omega_0_sq * d_a1_d_w1[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w1;
        T d_den_sq_d_w2 = T(2) * den_real * d_den_real_d_w2
                       + T(2) * a1 * omega_0_sq * d_a1_d_w2[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w2;

        // Second derivatives of den_real = a2 - omega_0²
        // d²(den_real)/dω² = d²a2/dω² - 2*(d_omega0/dω)² - 2*omega_0*d²_omega0/dω²
        bool is_middle = has_middle && (i == actual_n - 1);
        T d2_a1_d_w1_w1_i, d2_a1_d_w2_w2_i, d2_a1_d_w1_w2_i;
        T d2_a2_d_w1_w1_i, d2_a2_d_w2_w2_i, d2_a2_d_w1_w2_i;

        if (is_middle) {
            d2_a1_d_w1_w1_i = T(0); d2_a1_d_w2_w2_i = T(0); d2_a1_d_w1_w2_i = T(0);
            d2_a2_d_w1_w1_i = T(0); d2_a2_d_w2_w2_i = T(0); d2_a2_d_w1_w2_i = T(1);
        } else {
            // Use the Hessians computed earlier (need to recompute here)
            c10::complex<T> lp = poles[i].lp;
            c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
            c10::complex<T> half_Bp = poles[i].half_Bp;
            c10::complex<T> bp = poles[i].bp;
            bool is_bp1 = poles[i].is_bp1;

            c10::complex<T> lp_sq = lp * lp;
            c10::complex<T> d2_D_d_w1_w1 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w2_w2 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w1_w2 = -lp_sq / T(2) - c10::complex<T>(T(1), T(0));

            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);
            c10::complex<T> d_D_d_w1 = T(2) * half_Bp * d_half_Bp_d_w1 - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = T(2) * half_Bp * d_half_Bp_d_w2 - c10::complex<T>(omega_p1, T(0));

            c10::complex<T> D = sqrt_disc * sqrt_disc;
            T D_mag = std::abs(D);

            if (D_mag < std::numeric_limits<T>::epsilon() * T(100)) {
                d2_a1_d_w1_w1_i = T(0); d2_a1_d_w2_w2_i = T(0); d2_a1_d_w1_w2_i = T(0);
                d2_a2_d_w1_w1_i = T(0); d2_a2_d_w2_w2_i = T(0); d2_a2_d_w1_w2_i = T(0);
            } else {
                c10::complex<T> D_cubed_half = D * sqrt_disc;
                c10::complex<T> four_D_cubed_half = T(4) * D_cubed_half;

                c10::complex<T> d2_sqrt_d_w1_w1 = (T(2) * D * d2_D_d_w1_w1 - d_D_d_w1 * d_D_d_w1) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w2_w2 = (T(2) * D * d2_D_d_w2_w2 - d_D_d_w2 * d_D_d_w2) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w1_w2 = (T(2) * D * d2_D_d_w1_w2 - d_D_d_w1 * d_D_d_w2) / four_D_cubed_half;

                c10::complex<T> d2_bp_d_w1_w1, d2_bp_d_w2_w2, d2_bp_d_w1_w2;
                if (is_bp1) {
                    d2_bp_d_w1_w1 = d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = d2_sqrt_d_w1_w2;
                } else {
                    d2_bp_d_w1_w1 = -d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = -d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = -d2_sqrt_d_w1_w2;
                }

                c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
                c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
                c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

                c10::complex<T> d_bp_d_w1, d_bp_d_w2;
                if (is_bp1) {
                    d_bp_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
                    d_bp_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;
                } else {
                    d_bp_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
                    d_bp_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;
                }

                d2_a1_d_w1_w1_i = -T(2) * d2_bp_d_w1_w1.real();
                d2_a1_d_w2_w2_i = -T(2) * d2_bp_d_w2_w2.real();
                d2_a1_d_w1_w2_i = -T(2) * d2_bp_d_w1_w2.real();

                T d2_a2_contrib_11 = d_bp_d_w1.real() * d_bp_d_w1.real() + d_bp_d_w1.imag() * d_bp_d_w1.imag()
                                   + bp.real() * d2_bp_d_w1_w1.real() + bp.imag() * d2_bp_d_w1_w1.imag();
                T d2_a2_contrib_22 = d_bp_d_w2.real() * d_bp_d_w2.real() + d_bp_d_w2.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w2_w2.real() + bp.imag() * d2_bp_d_w2_w2.imag();
                T d2_a2_contrib_12 = d_bp_d_w1.real() * d_bp_d_w2.real() + d_bp_d_w1.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w1_w2.real() + bp.imag() * d2_bp_d_w1_w2.imag();

                d2_a2_d_w1_w1_i = T(2) * d2_a2_contrib_11;
                d2_a2_d_w2_w2_i = T(2) * d2_a2_contrib_22;
                d2_a2_d_w1_w2_i = T(2) * d2_a2_contrib_12;
            }
        }

        // Second derivatives of den_real = a2 - omega_0²
        T d2_den_real_d_w1_w1 = d2_a2_d_w1_w1_i - T(2) * d_omega0_d_w1 * d_omega0_d_w1 - T(2) * omega_0 * d2_omega0_d_w1_w1;
        T d2_den_real_d_w2_w2 = d2_a2_d_w2_w2_i - T(2) * d_omega0_d_w2 * d_omega0_d_w2 - T(2) * omega_0 * d2_omega0_d_w2_w2;
        T d2_den_real_d_w1_w2 = d2_a2_d_w1_w2_i - T(2) * d_omega0_d_w1 * d_omega0_d_w2 - T(2) * omega_0 * d2_omega0_d_w1_w2;

        // Second derivatives of den_sq = den_real² + a1² * omega_0²
        // d²(den_sq)/dω² = d²[(a2-ω₀²)² + a₁²ω₀²]/dω²
        T d2_den_sq_d_w1_w1 = T(2) * d_den_real_d_w1 * d_den_real_d_w1 + T(2) * den_real * d2_den_real_d_w1_w1
                           + T(2) * d_a1_d_w1[i] * d_a1_d_w1[i] * omega_0_sq + T(2) * a1 * omega_0_sq * d2_a1_d_w1_w1_i
                           + T(8) * a1 * omega_0 * d_a1_d_w1[i] * d_omega0_d_w1
                           + T(2) * a1 * a1 * d_omega0_d_w1 * d_omega0_d_w1 + T(2) * a1 * a1 * omega_0 * d2_omega0_d_w1_w1;

        T d2_den_sq_d_w2_w2 = T(2) * d_den_real_d_w2 * d_den_real_d_w2 + T(2) * den_real * d2_den_real_d_w2_w2
                           + T(2) * d_a1_d_w2[i] * d_a1_d_w2[i] * omega_0_sq + T(2) * a1 * omega_0_sq * d2_a1_d_w2_w2_i
                           + T(8) * a1 * omega_0 * d_a1_d_w2[i] * d_omega0_d_w2
                           + T(2) * a1 * a1 * d_omega0_d_w2 * d_omega0_d_w2 + T(2) * a1 * a1 * omega_0 * d2_omega0_d_w2_w2;

        T d2_den_sq_d_w1_w2 = T(2) * d_den_real_d_w1 * d_den_real_d_w2 + T(2) * den_real * d2_den_real_d_w1_w2
                           + T(2) * d_a1_d_w1[i] * d_a1_d_w2[i] * omega_0_sq + T(2) * a1 * omega_0_sq * d2_a1_d_w1_w2_i
                           + T(4) * a1 * omega_0 * (d_a1_d_w1[i] * d_omega0_d_w2 + d_a1_d_w2[i] * d_omega0_d_w1)
                           + T(2) * a1 * a1 * d_omega0_d_w1 * d_omega0_d_w2 + T(2) * a1 * a1 * omega_0 * d2_omega0_d_w1_w2;

        // d²log(den_sq)/dω² = d²(den_sq)/dω² / den_sq - (d(den_sq)/dω)² / den_sq²
        T d2_log_den_sq_d_w1_w1 = d2_den_sq_d_w1_w1 / den_sq - d_den_sq_d_w1 * d_den_sq_d_w1 / den_sq_sq;
        T d2_log_den_sq_d_w2_w2 = d2_den_sq_d_w2_w2 / den_sq - d_den_sq_d_w2 * d_den_sq_d_w2 / den_sq_sq;
        T d2_log_den_sq_d_w1_w2 = d2_den_sq_d_w1_w2 / den_sq - d_den_sq_d_w1 * d_den_sq_d_w2 / den_sq_sq;

        // Subtract from log(total_gain_sq) Hessian
        d2_log_gain_sq_d_w1_w1 -= d2_log_den_sq_d_w1_w1;
        d2_log_gain_sq_d_w2_w2 -= d2_log_den_sq_d_w2_w2;
        d2_log_gain_sq_d_w1_w2 -= d2_log_den_sq_d_w1_w2;
    }

    // Convert log(total_gain_sq) Hessian to nf Hessian
    // log(nf) = -1/(2n) * log(total_gain_sq)
    // nf = exp(log_nf)
    // d²nf/dω² = nf * [(d log_nf/dω)² + d² log_nf/dω²]
    T factor = T(-1) / (T(2) * T(actual_n));
    T d_log_nf_d_w1 = factor * d_log_gain_sq_d_w1;
    T d_log_nf_d_w2 = factor * d_log_gain_sq_d_w2;

    T d2_log_nf_d_w1_w1 = factor * d2_log_gain_sq_d_w1_w1;
    T d2_log_nf_d_w2_w2 = factor * d2_log_gain_sq_d_w2_w2;
    T d2_log_nf_d_w1_w2 = factor * d2_log_gain_sq_d_w1_w2;

    T d2_nf_d_w1_w1 = nf * (d_log_nf_d_w1 * d_log_nf_d_w1 + d2_log_nf_d_w1_w1);
    T d2_nf_d_w2_w2 = nf * (d_log_nf_d_w2 * d_log_nf_d_w2 + d2_log_nf_d_w2_w2);
    T d2_nf_d_w1_w2 = nf * (d_log_nf_d_w1 * d_log_nf_d_w2 + d2_log_nf_d_w1_w2);

    if (has_grad_grad_omega_p1) {
        new_grad_omega_p1 += total_grad_b0 * d2_nf_d_w1_w1 * grad_grad_omega_p1;
        new_grad_omega_p2 += total_grad_b0 * d2_nf_d_w1_w2 * grad_grad_omega_p1;
    }
    if (has_grad_grad_omega_p2) {
        new_grad_omega_p1 += total_grad_b0 * d2_nf_d_w1_w2 * grad_grad_omega_p2;
        new_grad_omega_p2 += total_grad_b0 * d2_nf_d_w2_w2 * grad_grad_omega_p2;
    }
}

}  // anonymous namespace

/**
 * CPU implementation of butterworth_analog_bandpass_filter.
 *
 * This is NOT an element-wise operation: output shape is (*batch_shape, n, 6)
 * where n is the filter order.
 *
 * @param n Filter order (positive integer)
 * @param omega_p1 Lower passband frequency tensor, shape (*batch_shape)
 * @param omega_p2 Upper passband frequency tensor, shape (*batch_shape)
 * @return SOS coefficients tensor, shape (*batch_shape, n, 6)
 */
inline at::Tensor butterworth_analog_bandpass_filter(
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    TORCH_CHECK(n > 0, "butterworth_analog_bandpass_filter: order n must be positive, got ", n);
    TORCH_CHECK(n <= 64, "butterworth_analog_bandpass_filter: order n must be <= 64, got ", n);

    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();

    // Compute output shape: (*batch_shape, n, 6)
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(n);
    output_shape.push_back(6);

    // Create output tensor
    auto options = omega_p1_bc.options();
    at::Tensor output = at::empty(output_shape, options);

    // Flatten batch dimensions for parallel processing
    int64_t batch_size = omega_p1_bc.numel();
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();
    at::Tensor output_flat = output.view({batch_size, n, 6});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_cpu",
        [&]() {
            const scalar_t* omega_p1_data = omega_p1_flat.data_ptr<scalar_t>();
            const scalar_t* omega_p2_data = omega_p2_flat.data_ptr<scalar_t>();
            scalar_t* output_data = output_flat.data_ptr<scalar_t>();

            // Parallel over batch dimension
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    scalar_t w1 = omega_p1_data[i];
                    scalar_t w2 = omega_p2_data[i];

                    // Validate frequencies
                    // Allow any positive frequencies where w1 < w2
                    // (normalized to Nyquist is handled at Python level)

                    // Compute SOS for this batch element
                    scalar_t* sos_ptr = output_data + i * n * 6;

                    // Use float for computation if half precision
                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        float w1_f = static_cast<float>(w1);
                        float w2_f = static_cast<float>(w2);
                        float sos_f[64 * 6];  // Max order * 6

                        butterworth_analog_bandpass_filter_kernel<float>(
                            n, w1_f, w2_f, sos_f
                        );

                        // Convert back to scalar_t
                        for (int64_t j = 0; j < n * 6; ++j) {
                            sos_ptr[j] = static_cast<scalar_t>(sos_f[j]);
                        }
                    } else {
                        butterworth_analog_bandpass_filter_kernel<scalar_t>(
                            n, w1, w2, sos_ptr
                        );
                    }
                }
            });
        }
    );

    return output;
}

/**
 * Backward pass for butterworth_analog_bandpass_filter on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward(
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();
    int64_t batch_size = omega_p1_bc.numel();

    // Flatten for processing
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();

    // grad_output shape: (*batch_shape, n, 6)
    at::Tensor grad_output_flat = grad_output.view({batch_size, n, 6}).contiguous();

    // Create output gradients
    at::Tensor grad_omega_p1_flat = at::empty({batch_size}, omega_p1_flat.options());
    at::Tensor grad_omega_p2_flat = at::empty({batch_size}, omega_p2_flat.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_backward_cpu",
        [&]() {
            const scalar_t* omega_p1_data = omega_p1_flat.data_ptr<scalar_t>();
            const scalar_t* omega_p2_data = omega_p2_flat.data_ptr<scalar_t>();
            const scalar_t* grad_output_data = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_omega_p1_data = grad_omega_p1_flat.data_ptr<scalar_t>();
            scalar_t* grad_omega_p2_data = grad_omega_p2_flat.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    scalar_t w1 = omega_p1_data[i];
                    scalar_t w2 = omega_p2_data[i];
                    const scalar_t* grad_sos = grad_output_data + i * n * 6;

                    scalar_t grad_w1, grad_w2;

                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        float w1_f = static_cast<float>(w1);
                        float w2_f = static_cast<float>(w2);
                        float grad_sos_f[64 * 6];
                        for (int64_t j = 0; j < n * 6; ++j) {
                            grad_sos_f[j] = static_cast<float>(grad_sos[j]);
                        }
                        float grad_w1_f, grad_w2_f;

                        butterworth_analog_bandpass_filter_backward_kernel<float>(
                            grad_sos_f, n, w1_f, w2_f, grad_w1_f, grad_w2_f
                        );

                        grad_w1 = static_cast<scalar_t>(grad_w1_f);
                        grad_w2 = static_cast<scalar_t>(grad_w2_f);
                    } else {
                        butterworth_analog_bandpass_filter_backward_kernel<scalar_t>(
                            grad_sos, n, w1, w2, grad_w1, grad_w2
                        );
                    }

                    grad_omega_p1_data[i] = grad_w1;
                    grad_omega_p2_data[i] = grad_w2;
                }
            });
        }
    );

    // Reshape back to original batch shape
    at::Tensor grad_omega_p1 = grad_omega_p1_flat.view(batch_shape);
    at::Tensor grad_omega_p2 = grad_omega_p2_flat.view(batch_shape);

    return std::make_tuple(grad_omega_p1, grad_omega_p2);
}

/**
 * Double-backward pass for butterworth_analog_bandpass_filter on CPU.
 *
 * Computes second-order gradients for Hessian-vector products.
 *
 * @param grad_grad_omega_p1 Gradient w.r.t. grad_omega_p1 from first backward
 * @param grad_grad_omega_p2 Gradient w.r.t. grad_omega_p2 from first backward
 * @param grad_output Original gradient w.r.t. SOS output
 * @param n Filter order
 * @param omega_p1 Lower passband frequency
 * @param omega_p2 Upper passband frequency
 * @return Tuple of (grad_grad_output, new_grad_omega_p1, new_grad_omega_p2)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward_backward(
    const at::Tensor& grad_grad_omega_p1,
    const at::Tensor& grad_grad_omega_p2,
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();
    int64_t batch_size = omega_p1_bc.numel();

    // Flatten for processing
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();

    // grad_output shape: (*batch_shape, n, 6)
    at::Tensor grad_output_flat = grad_output.view({batch_size, n, 6}).contiguous();

    // Handle optional grad_grad inputs
    bool has_gg_omega_p1 = grad_grad_omega_p1.defined() && grad_grad_omega_p1.numel() > 0;
    bool has_gg_omega_p2 = grad_grad_omega_p2.defined() && grad_grad_omega_p2.numel() > 0;

    at::Tensor gg_omega_p1_flat, gg_omega_p2_flat;
    if (has_gg_omega_p1) {
        gg_omega_p1_flat = grad_grad_omega_p1.flatten().contiguous();
    }
    if (has_gg_omega_p2) {
        gg_omega_p2_flat = grad_grad_omega_p2.flatten().contiguous();
    }

    // Create outputs
    at::Tensor grad_grad_output_flat = at::zeros({batch_size, n, 6}, grad_output.options());
    at::Tensor new_grad_omega_p1_flat = at::zeros({batch_size}, omega_p1_flat.options());
    at::Tensor new_grad_omega_p2_flat = at::zeros({batch_size}, omega_p2_flat.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_backward_backward_cpu",
        [&]() {
            const scalar_t* omega_p1_data = omega_p1_flat.data_ptr<scalar_t>();
            const scalar_t* omega_p2_data = omega_p2_flat.data_ptr<scalar_t>();
            const scalar_t* grad_output_data = grad_output_flat.data_ptr<scalar_t>();

            const scalar_t* gg_omega_p1_data = has_gg_omega_p1 ? gg_omega_p1_flat.data_ptr<scalar_t>() : nullptr;
            const scalar_t* gg_omega_p2_data = has_gg_omega_p2 ? gg_omega_p2_flat.data_ptr<scalar_t>() : nullptr;

            scalar_t* grad_grad_output_data = grad_grad_output_flat.data_ptr<scalar_t>();
            scalar_t* new_grad_omega_p1_data = new_grad_omega_p1_flat.data_ptr<scalar_t>();
            scalar_t* new_grad_omega_p2_data = new_grad_omega_p2_flat.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    scalar_t w1 = omega_p1_data[i];
                    scalar_t w2 = omega_p2_data[i];
                    const scalar_t* grad_sos = grad_output_data + i * n * 6;

                    scalar_t gg_w1 = has_gg_omega_p1 ? gg_omega_p1_data[i] : scalar_t(0);
                    scalar_t gg_w2 = has_gg_omega_p2 ? gg_omega_p2_data[i] : scalar_t(0);

                    scalar_t* grad_grad_sos = grad_grad_output_data + i * n * 6;
                    scalar_t new_grad_w1, new_grad_w2;

                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        float w1_f = static_cast<float>(w1);
                        float w2_f = static_cast<float>(w2);
                        float gg_w1_f = static_cast<float>(gg_w1);
                        float gg_w2_f = static_cast<float>(gg_w2);

                        float grad_sos_f[64 * 6];
                        float grad_grad_sos_f[64 * 6];
                        for (int64_t j = 0; j < n * 6; ++j) {
                            grad_sos_f[j] = static_cast<float>(grad_sos[j]);
                        }

                        float new_grad_w1_f, new_grad_w2_f;

                        butterworth_analog_bandpass_filter_backward_backward_kernel<float>(
                            gg_w1_f,
                            gg_w2_f,
                            grad_sos_f,
                            n,
                            w1_f,
                            w2_f,
                            has_gg_omega_p1,
                            has_gg_omega_p2,
                            grad_grad_sos_f,
                            new_grad_w1_f,
                            new_grad_w2_f
                        );

                        for (int64_t j = 0; j < n * 6; ++j) {
                            grad_grad_sos[j] = static_cast<scalar_t>(grad_grad_sos_f[j]);
                        }
                        new_grad_w1 = static_cast<scalar_t>(new_grad_w1_f);
                        new_grad_w2 = static_cast<scalar_t>(new_grad_w2_f);
                    } else {
                        butterworth_analog_bandpass_filter_backward_backward_kernel<scalar_t>(
                            gg_w1,
                            gg_w2,
                            grad_sos,
                            n,
                            w1,
                            w2,
                            has_gg_omega_p1,
                            has_gg_omega_p2,
                            grad_grad_sos,
                            new_grad_w1,
                            new_grad_w2
                        );
                    }

                    new_grad_omega_p1_data[i] = new_grad_w1;
                    new_grad_omega_p2_data[i] = new_grad_w2;
                }
            });
        }
    );

    // Reshape back to original shapes
    at::Tensor grad_grad_output_reshaped = grad_grad_output_flat.view(grad_output.sizes());
    at::Tensor new_grad_omega_p1 = new_grad_omega_p1_flat.view(batch_shape);
    at::Tensor new_grad_omega_p2 = new_grad_omega_p2_flat.view(batch_shape);

    return std::make_tuple(grad_grad_output_reshaped, new_grad_omega_p1, new_grad_omega_p2);
}

}  // namespace torchscience::cpu::filter

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "butterworth_analog_bandpass_filter",
        &torchscience::cpu::filter::butterworth_analog_bandpass_filter
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward",
        &torchscience::cpu::filter::butterworth_analog_bandpass_filter_backward
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward_backward",
        &torchscience::cpu::filter::butterworth_analog_bandpass_filter_backward_backward
    );
}
