#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/gamma.h"
#include "impl/special_functions/log_gamma.h"

using namespace torchscience::impl::special_functions;

class LogGammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(LogGammaTest, ComplexFloat_PositiveIntegers) {
  // log(Gamma(n)) = log((n-1)!)
  c10::complex<float> z1(1.0f, 0.0f);
  auto result1 = log_gamma_complex(z1);
  EXPECT_NEAR(result1.real(), 0.0f, 1e-5f);  // log(1) = 0
  EXPECT_NEAR(result1.imag(), 0.0f, 1e-6f);

  c10::complex<float> z2(2.0f, 0.0f);
  auto result2 = log_gamma_complex(z2);
  EXPECT_NEAR(result2.real(), 0.0f, 1e-5f);  // log(1) = 0
  EXPECT_NEAR(result2.imag(), 0.0f, 1e-6f);

  c10::complex<float> z3(3.0f, 0.0f);
  auto result3 = log_gamma_complex(z3);
  EXPECT_NEAR(result3.real(), std::log(2.0f), 1e-5f);  // log(2!)
  EXPECT_NEAR(result3.imag(), 0.0f, 1e-6f);

  c10::complex<float> z4(4.0f, 0.0f);
  auto result4 = log_gamma_complex(z4);
  EXPECT_NEAR(result4.real(), std::log(6.0f), 1e-5f);  // log(3!)
  EXPECT_NEAR(result4.imag(), 0.0f, 1e-6f);
}

TEST_F(LogGammaTest, ComplexFloat_HalfInteger) {
  // log(Gamma(1/2)) = log(sqrt(pi)) = 0.5 * log(pi)
  c10::complex<float> z(0.5f, 0.0f);
  auto result = log_gamma_complex(z);
  float expected = 0.5f * std::log(static_cast<float>(kPi));
  EXPECT_NEAR(result.real(), expected, 1e-5f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(LogGammaTest, ComplexFloat_Pole) {
  // At poles, should return infinity
  c10::complex<float> z(0.0f, 0.0f);
  auto result = log_gamma_complex(z);
  EXPECT_TRUE(std::isinf(result.real()));
}

TEST_F(LogGammaTest, ComplexFloat_General) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = log_gamma_complex(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(LogGammaTest, ComplexDouble_PositiveIntegers) {
  c10::complex<double> z1(1.0, 0.0);
  auto result1 = log_gamma_complex(z1);
  EXPECT_NEAR(result1.real(), 0.0, 1e-14);
  EXPECT_NEAR(result1.imag(), 0.0, 1e-14);

  c10::complex<double> z2(2.0, 0.0);
  auto result2 = log_gamma_complex(z2);
  EXPECT_NEAR(result2.real(), 0.0, 1e-14);
  EXPECT_NEAR(result2.imag(), 0.0, 1e-14);

  c10::complex<double> z3(3.0, 0.0);
  auto result3 = log_gamma_complex(z3);
  EXPECT_NEAR(result3.real(), std::log(2.0), 1e-14);
  EXPECT_NEAR(result3.imag(), 0.0, 1e-14);

  c10::complex<double> z4(4.0, 0.0);
  auto result4 = log_gamma_complex(z4);
  EXPECT_NEAR(result4.real(), std::log(6.0), 1e-14);
  EXPECT_NEAR(result4.imag(), 0.0, 1e-14);

  c10::complex<double> z5(5.0, 0.0);
  auto result5 = log_gamma_complex(z5);
  EXPECT_NEAR(result5.real(), std::log(24.0), 1e-13);
  EXPECT_NEAR(result5.imag(), 0.0, 1e-14);
}

TEST_F(LogGammaTest, ComplexDouble_HalfInteger) {
  c10::complex<double> z(0.5, 0.0);
  auto result = log_gamma_complex(z);
  double expected = 0.5 * std::log(kPi);
  EXPECT_NEAR(result.real(), expected, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(LogGammaTest, ComplexDouble_ThreeHalves) {
  // log(Gamma(3/2)) = log(sqrt(pi)/2) = 0.5*log(pi) - log(2)
  c10::complex<double> z(1.5, 0.0);
  auto result = log_gamma_complex(z);
  double expected = 0.5 * std::log(kPi) - std::log(2.0);
  EXPECT_NEAR(result.real(), expected, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(LogGammaTest, ComplexDouble_Pole) {
  c10::complex<double> z(0.0, 0.0);
  auto result = log_gamma_complex(z);
  EXPECT_TRUE(std::isinf(result.real()));
}

TEST_F(LogGammaTest, ComplexDouble_NegativeIntegerPole) {
  c10::complex<double> z(-1.0, 0.0);
  auto result = log_gamma_complex(z);
  EXPECT_TRUE(std::isinf(result.real()));

  c10::complex<double> z2(-5.0, 0.0);
  auto result2 = log_gamma_complex(z2);
  EXPECT_TRUE(std::isinf(result2.real()));
}

TEST_F(LogGammaTest, ComplexDouble_General) {
  c10::complex<double> z(2.0, 1.0);
  auto result = log_gamma_complex(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

TEST_F(LogGammaTest, ComplexDouble_LargePositive) {
  // For large z, log(Gamma(z)) should be approximately (z-0.5)*log(z) - z + 0.5*log(2*pi)
  c10::complex<double> z(100.0, 0.0);
  auto result = log_gamma_complex(z);
  // Stirling's approximation
  double stirling_approx = (100.0 - 0.5) * std::log(100.0) - 100.0 + 0.5 * std::log(2.0 * kPi);
  EXPECT_NEAR(result.real(), stirling_approx, 1.0);  // Within 1 of Stirling
  EXPECT_NEAR(result.imag(), 0.0, 1e-12);
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(LogGammaTest, ComplexDouble_ExponentialConsistency) {
  // exp(log_gamma(z)) should equal gamma(z) for moderate z
  std::vector<c10::complex<double>> test_values = {
      c10::complex<double>(1.5, 0.0),
      c10::complex<double>(2.5, 0.0),
      c10::complex<double>(3.0, 0.0),
      c10::complex<double>(2.0, 0.5)
  };

  for (const auto& z : test_values) {
    auto log_result = log_gamma_complex(z);
    auto exp_log_result = std::exp(log_result);
    auto direct_result = gamma(z);

    EXPECT_NEAR(exp_log_result.real(), direct_result.real(),
                std::abs(direct_result.real()) * 1e-10)
        << "Real part mismatch for z = " << z.real() << " + " << z.imag() << "i";
    EXPECT_NEAR(exp_log_result.imag(), direct_result.imag(),
                std::abs(direct_result.imag()) * 1e-10 + 1e-14)
        << "Imag part mismatch for z = " << z.real() << " + " << z.imag() << "i";
  }
}

// ============================================================================
// Reflection Formula Tests
// ============================================================================

TEST_F(LogGammaTest, ComplexDouble_ReflectionFormula) {
  // log(Gamma(z)) + log(Gamma(1-z)) = log(pi) - log(sin(pi*z))
  // This is tricky due to branch cuts, so we just verify the sum of lgamma values
  std::vector<double> test_values = {0.1, 0.25, 0.3, 0.4};
  for (double x : test_values) {
    c10::complex<double> z(x, 0.0);
    c10::complex<double> one_minus_z(1.0 - x, 0.0);

    auto lg_z = log_gamma_complex(z);
    auto lg_one_minus_z = log_gamma_complex(one_minus_z);

    // exp(lg_z) * exp(lg_one_minus_z) should equal pi / sin(pi*x)
    double product_real = std::exp(lg_z.real() + lg_one_minus_z.real());
    double expected = kPi / std::sin(kPi * x);

    EXPECT_NEAR(product_real, expected, expected * 1e-10)
        << "Failed for x = " << x;
  }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(LogGammaTest, ComplexDouble_NoOverflow) {
  // log_gamma should not overflow even for large arguments
  c10::complex<double> z(500.0, 0.0);
  auto result = log_gamma_complex(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isinf(result.real()));

  c10::complex<double> z2(1000.0, 0.0);
  auto result2 = log_gamma_complex(z2);
  EXPECT_FALSE(std::isnan(result2.real()));
  EXPECT_FALSE(std::isinf(result2.real()));
}
