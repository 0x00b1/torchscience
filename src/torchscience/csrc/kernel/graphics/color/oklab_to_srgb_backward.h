#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for Oklab to sRGB conversion.
 *
 * Computes gradients w.r.t. Oklab input given gradients w.r.t. sRGB output.
 * Uses chain rule through the composition: Lab -> L'M'S' -> LMS -> linear RGB -> sRGB
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param lab Original Oklab input [L, a, b]
 * @param grad_lab Output gradient w.r.t. Oklab [dL, da, db]
 */
template <typename T>
void oklab_to_srgb_backward_scalar(const T* grad_rgb, const T* lab, T* grad_lab) {
    // sRGB encoding constants
    const T threshold_linear = T(0.0031308);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    const T L = lab[0];
    const T a = lab[1];
    const T b = lab[2];

    // Forward pass to get intermediate values
    // Inverse M2 matrix (Oklab to L'M'S')
    const T l_prime = L + T(0.3963377774) * a + T(0.2158037573) * b;
    const T m_prime = L - T(0.1055613458) * a - T(0.0638541728) * b;
    const T s_prime = L - T(0.0894841775) * a - T(1.2914855480) * b;

    // Cube to get LMS
    const T l = l_prime * l_prime * l_prime;
    const T m = m_prime * m_prime * m_prime;
    const T s = s_prime * s_prime * s_prime;

    // Inverse M1 matrix (LMS to linear RGB)
    const T r_linear = T(+4.0767416621) * l - T(3.3077115913) * m + T(0.2309699292) * s;
    const T g_linear = T(-1.2684380046) * l + T(2.6097574011) * m - T(0.3413193965) * s;
    const T b_linear = T(-0.0041960863) * l - T(0.7034186147) * m + T(1.7076147010) * s;

    // Chain rule: sRGB -> linear RGB
    // Derivative of sRGB encoding
    T linear[3] = {r_linear, g_linear, b_linear};
    T grad_linear[3];
    for (int i = 0; i < 3; ++i) {
        const T v = linear[i];
        T deriv;
        if (v <= threshold_linear) {
            deriv = linear_slope;
        } else if (v > T(0)) {
            // d/dx [scale * x^inv_gamma - offset] = scale * inv_gamma * x^(inv_gamma-1)
            deriv = scale * inv_gamma * std::pow(v, inv_gamma - T(1));
        } else {
            // d/dx [-scale * (-x)^inv_gamma + offset] = scale * inv_gamma * (-x)^(inv_gamma-1)
            deriv = scale * inv_gamma * std::pow(-v, inv_gamma - T(1));
        }
        grad_linear[i] = grad_rgb[i] * deriv;
    }

    // Chain rule: linear RGB -> LMS (transpose of inverse M1)
    // inv_M1 = [[4.0767416621, -3.3077115913, 0.2309699292],
    //           [-1.2684380046, 2.6097574011, -0.3413193965],
    //           [-0.0041960863, -0.7034186147, 1.7076147010]]
    const T grad_l = T(+4.0767416621) * grad_linear[0] - T(1.2684380046) * grad_linear[1] - T(0.0041960863) * grad_linear[2];
    const T grad_m = -T(3.3077115913) * grad_linear[0] + T(2.6097574011) * grad_linear[1] - T(0.7034186147) * grad_linear[2];
    const T grad_s = T(0.2309699292) * grad_linear[0] - T(0.3413193965) * grad_linear[1] + T(1.7076147010) * grad_linear[2];

    // Chain rule: LMS -> L'M'S'
    // Derivative of x^3 = 3x^2
    const T grad_l_prime = grad_l * T(3) * l_prime * l_prime;
    const T grad_m_prime = grad_m * T(3) * m_prime * m_prime;
    const T grad_s_prime = grad_s * T(3) * s_prime * s_prime;

    // Chain rule: L'M'S' -> Lab (transpose of inverse M2)
    // inv_M2 = [[1, 0.3963377774, 0.2158037573],
    //           [1, -0.1055613458, -0.0638541728],
    //           [1, -0.0894841775, -1.2914855480]]
    grad_lab[0] = grad_l_prime + grad_m_prime + grad_s_prime;
    grad_lab[1] = T(0.3963377774) * grad_l_prime - T(0.1055613458) * grad_m_prime - T(0.0894841775) * grad_s_prime;
    grad_lab[2] = T(0.2158037573) * grad_l_prime - T(0.0638541728) * grad_m_prime - T(1.2914855480) * grad_s_prime;
}

}  // namespace torchscience::kernel::graphics::color
