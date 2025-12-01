#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/lanczos_approximation.h"

using namespace torchscience::impl::special_functions;

class LanczosApproximationTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Constants Tests
// ============================================================================

TEST_F(LanczosApproximationTest, Constants_LanczosG) {
  EXPECT_DOUBLE_EQ(kLanczosG, 7.0);
}

TEST_F(LanczosApproximationTest, Constants_LanczosN) {
  EXPECT_EQ(kLanczosN, 9);
}

TEST_F(LanczosApproximationTest, Constants_Sqrt2Pi) {
  double expected = std::sqrt(2.0 * kPi);
  EXPECT_NEAR(kSqrt2Pi, expected, 1e-15);
}

TEST_F(LanczosApproximationTest, Constants_FirstCoefficient) {
  // First coefficient should be approximately 1
  EXPECT_NEAR(kLanczosCoeffs[0], 1.0, 1e-10);
}

// ============================================================================
// Float Lanczos Series Tests
// ============================================================================

TEST_F(LanczosApproximationTest, Float_LanczosSeries_One) {
  float result = lanczos_series(1.0f);
  // The series evaluated at z=1 should give a specific value
  // This is used in Gamma(1) = 1
  EXPECT_GT(result, 0.0f);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

TEST_F(LanczosApproximationTest, Float_LanczosSeries_SmallPositive) {
  std::vector<float> test_values = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f};
  for (float z : test_values) {
    float result = lanczos_series(z);
    EXPECT_GT(result, 0.0f) << "Failed for z = " << z;
    EXPECT_FALSE(std::isnan(result)) << "Failed for z = " << z;
    EXPECT_FALSE(std::isinf(result)) << "Failed for z = " << z;
  }
}

TEST_F(LanczosApproximationTest, Float_LanczosSeries_LargeValues) {
  std::vector<float> test_values = {10.0f, 50.0f, 100.0f};
  for (float z : test_values) {
    float result = lanczos_series(z);
    EXPECT_GT(result, 0.0f) << "Failed for z = " << z;
    EXPECT_FALSE(std::isnan(result)) << "Failed for z = " << z;
    EXPECT_FALSE(std::isinf(result)) << "Failed for z = " << z;
  }
}

// ============================================================================
// Double Lanczos Series Tests
// ============================================================================

TEST_F(LanczosApproximationTest, Double_LanczosSeries_One) {
  double result = lanczos_series(1.0);
  EXPECT_GT(result, 0.0);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

TEST_F(LanczosApproximationTest, Double_LanczosSeries_SmallPositive) {
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0};
  for (double z : test_values) {
    double result = lanczos_series(z);
    EXPECT_GT(result, 0.0) << "Failed for z = " << z;
    EXPECT_FALSE(std::isnan(result)) << "Failed for z = " << z;
    EXPECT_FALSE(std::isinf(result)) << "Failed for z = " << z;
  }
}

TEST_F(LanczosApproximationTest, Double_LanczosSeries_LargeValues) {
  std::vector<double> test_values = {10.0, 50.0, 100.0, 500.0};
  for (double z : test_values) {
    double result = lanczos_series(z);
    EXPECT_GT(result, 0.0) << "Failed for z = " << z;
    EXPECT_FALSE(std::isnan(result)) << "Failed for z = " << z;
    EXPECT_FALSE(std::isinf(result)) << "Failed for z = " << z;
  }
}

// ============================================================================
// Complex Float Lanczos Series Tests
// ============================================================================

TEST_F(LanczosApproximationTest, ComplexFloat_LanczosSeries_RealAxis) {
  c10::complex<float> z(1.0f, 0.0f);
  auto result = lanczos_series(z);
  EXPECT_GT(result.real(), 0.0f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(LanczosApproximationTest, ComplexFloat_LanczosSeries_General) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = lanczos_series(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
  EXPECT_FALSE(std::isinf(result.real()));
  EXPECT_FALSE(std::isinf(result.imag()));
}

TEST_F(LanczosApproximationTest, ComplexFloat_LanczosSeries_ImaginaryAxis) {
  c10::complex<float> z(1.0f, 2.0f);
  auto result = lanczos_series(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

// ============================================================================
// Complex Double Lanczos Series Tests
// ============================================================================

TEST_F(LanczosApproximationTest, ComplexDouble_LanczosSeries_RealAxis) {
  c10::complex<double> z(1.0, 0.0);
  auto result = lanczos_series(z);
  // On real axis, should match real result
  double real_result = lanczos_series(1.0);
  EXPECT_NEAR(result.real(), real_result, 1e-12);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(LanczosApproximationTest, ComplexDouble_LanczosSeries_General) {
  c10::complex<double> z(2.0, 1.0);
  auto result = lanczos_series(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
  EXPECT_FALSE(std::isinf(result.real()));
  EXPECT_FALSE(std::isinf(result.imag()));
}

TEST_F(LanczosApproximationTest, ComplexDouble_LanczosSeries_ConjugateSymmetry) {
  // For real coefficients, A(z*) = A(z)*
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> z_conj(2.0, -1.0);

  auto result = lanczos_series(z);
  auto result_conj = lanczos_series(z_conj);

  EXPECT_NEAR(result.real(), result_conj.real(), 1e-14);
  EXPECT_NEAR(result.imag(), -result_conj.imag(), 1e-14);
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(LanczosApproximationTest, FloatDoubleConsistency) {
  // Float and double should give similar results
  std::vector<double> test_values = {1.0, 2.0, 3.0, 5.0, 10.0};
  for (double z : test_values) {
    float result_float = lanczos_series(static_cast<float>(z));
    double result_double = lanczos_series(z);
    EXPECT_NEAR(result_float, result_double, 1e-4)
        << "Inconsistency at z = " << z;
  }
}

TEST_F(LanczosApproximationTest, RealComplexConsistency) {
  // Complex on real axis should match real result
  std::vector<double> test_values = {1.0, 2.0, 3.0, 5.0, 10.0};
  for (double z : test_values) {
    double real_result = lanczos_series(z);
    c10::complex<double> z_complex(z, 0.0);
    auto complex_result = lanczos_series(z_complex);

    EXPECT_NEAR(complex_result.real(), real_result, 1e-12)
        << "Inconsistency at z = " << z;
    EXPECT_NEAR(complex_result.imag(), 0.0, 1e-14)
        << "Non-zero imaginary at z = " << z;
  }
}

// ============================================================================
// Smoothness Tests
// ============================================================================

TEST_F(LanczosApproximationTest, Double_Smoothness) {
  // Check that the series is smooth (small changes in input -> small changes in output)
  double z = 5.0;
  double delta = 1e-6;

  double f_z = lanczos_series(z);
  double f_z_plus = lanczos_series(z + delta);
  double f_z_minus = lanczos_series(z - delta);

  // Finite difference approximation to derivative
  double derivative_approx = (f_z_plus - f_z_minus) / (2.0 * delta);

  // The derivative should be finite and reasonably bounded
  EXPECT_FALSE(std::isnan(derivative_approx));
  EXPECT_FALSE(std::isinf(derivative_approx));
  EXPECT_LT(std::abs(derivative_approx), 100.0);  // Series has bounded derivative
}
