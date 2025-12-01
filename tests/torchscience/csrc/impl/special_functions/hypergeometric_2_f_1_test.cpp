#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/hypergeometric_2_f_1.h"

using namespace torchscience::impl::special_functions;

class Hypergeometric2F1Test : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Float Tests - Special Cases
// ============================================================================

TEST_F(Hypergeometric2F1Test, Float_ZeroArgument) {
  // 2F1(a, b; c; 0) = 1 for all valid a, b, c
  EXPECT_NEAR(hypergeometric_2_f_1(1.0f, 2.0f, 3.0f, 0.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(hypergeometric_2_f_1(0.5f, 0.5f, 1.0f, 0.0f), 1.0f, 1e-6f);
}

TEST_F(Hypergeometric2F1Test, Float_TerminatingA) {
  // If a is a non-positive integer, series terminates
  // 2F1(0, b; c; z) = 1
  EXPECT_NEAR(hypergeometric_2_f_1(0.0f, 1.0f, 2.0f, 0.5f), 1.0f, 1e-6f);

  // 2F1(-1, b; c; z) = 1 - bz/c
  float expected = 1.0f - 2.0f * 0.5f / 3.0f;
  EXPECT_NEAR(hypergeometric_2_f_1(-1.0f, 2.0f, 3.0f, 0.5f), expected, 1e-5f);
}

TEST_F(Hypergeometric2F1Test, Float_SmallZ) {
  // For small z, direct series should converge quickly
  float result = hypergeometric_2_f_1(1.0f, 1.0f, 2.0f, 0.1f);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Double Tests - Special Cases
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_ZeroArgument) {
  EXPECT_NEAR(hypergeometric_2_f_1(1.0, 2.0, 3.0, 0.0), 1.0, 1e-14);
  EXPECT_NEAR(hypergeometric_2_f_1(0.5, 0.5, 1.0, 0.0), 1.0, 1e-14);
}

TEST_F(Hypergeometric2F1Test, Double_TerminatingA) {
  // 2F1(0, b; c; z) = 1
  EXPECT_NEAR(hypergeometric_2_f_1(0.0, 1.0, 2.0, 0.5), 1.0, 1e-14);

  // 2F1(-1, b; c; z) = 1 - bz/c
  double expected = 1.0 - 2.0 * 0.5 / 3.0;
  EXPECT_NEAR(hypergeometric_2_f_1(-1.0, 2.0, 3.0, 0.5), expected, 1e-14);

  // 2F1(-2, b; c; z) = 1 - 2bz/c + b(b+1)z^2/(2c(c+1))
  double a = -2.0, b = 1.0, c = 2.0, z = 0.3;
  double term1 = 1.0;
  double term2 = a * b * z / c;
  double term3 = a * (a + 1) * b * (b + 1) * z * z / (2.0 * c * (c + 1));
  expected = term1 + term2 + term3;
  EXPECT_NEAR(hypergeometric_2_f_1(a, b, c, z), expected, 1e-12);
}

// ============================================================================
// Known Values Tests
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_KnownValue_Log) {
  // 2F1(1, 1; 2; z) = -log(1-z)/z
  double z = 0.5;
  double expected = -std::log(1.0 - z) / z;
  EXPECT_NEAR(hypergeometric_2_f_1(1.0, 1.0, 2.0, z), expected, 1e-12);
}

TEST_F(Hypergeometric2F1Test, Double_KnownValue_Arcsin) {
  // 2F1(1/2, 1/2; 3/2; z^2) = arcsin(z)/z for |z| < 1
  double z = 0.5;
  double z2 = z * z;
  double expected = std::asin(z) / z;
  EXPECT_NEAR(hypergeometric_2_f_1(0.5, 0.5, 1.5, z2), expected, 1e-10);
}

TEST_F(Hypergeometric2F1Test, Double_KnownValue_Sqrt) {
  // 2F1(-1/2, 1; 1; z) = sqrt(1-z)
  double z = 0.5;
  double expected = std::sqrt(1.0 - z);
  EXPECT_NEAR(hypergeometric_2_f_1(-0.5, 1.0, 1.0, z), expected, 1e-10);
}

// ============================================================================
// Gauss Summation Theorem (z = 1)
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_GaussSummation) {
  // 2F1(a, b; c; 1) = Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b))
  // Valid when Re(c - a - b) > 0
  double a = 0.5, b = 0.5, c = 2.0;
  // c - a - b = 2 - 0.5 - 0.5 = 1 > 0, so this converges
  double result = hypergeometric_2_f_1(a, b, c, 1.0);
  double expected = std::tgamma(c) * std::tgamma(c - a - b) /
                    (std::tgamma(c - a) * std::tgamma(c - b));
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(Hypergeometric2F1Test, Double_GaussSummation_Divergent) {
  // When Re(c - a - b) <= 0, 2F1(a,b;c;1) diverges
  double a = 1.0, b = 1.0, c = 1.5;
  // c - a - b = 1.5 - 1 - 1 = -0.5 < 0
  double result = hypergeometric_2_f_1(a, b, c, 1.0);
  EXPECT_TRUE(std::isnan(result) || std::isinf(result));
}

// ============================================================================
// Transformation Tests (z near 1)
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_ZNearOne) {
  // For z close to 1, should use 1-z transformation
  double a = 0.5, b = 0.5, c = 1.5;
  double z = 0.9;
  double result = hypergeometric_2_f_1(a, b, c, z);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
  EXPECT_GT(result, 0.0);  // Should be positive for these parameters
}

TEST_F(Hypergeometric2F1Test, Double_ZNearOneSmall) {
  double a = 0.25, b = 0.25, c = 1.0;
  double z = 0.8;
  double result = hypergeometric_2_f_1(a, b, c, z);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Complex Tests
// ============================================================================

TEST_F(Hypergeometric2F1Test, ComplexFloat_ZeroArgument) {
  c10::complex<float> a(1.0f, 0.0f);
  c10::complex<float> b(2.0f, 0.0f);
  c10::complex<float> c(3.0f, 0.0f);
  c10::complex<float> z(0.0f, 0.0f);

  auto result = hypergeometric_2_f_1(a, b, c, z);
  EXPECT_NEAR(result.real(), 1.0f, 1e-6f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(Hypergeometric2F1Test, ComplexDouble_ZeroArgument) {
  c10::complex<double> a(1.0, 0.0);
  c10::complex<double> b(2.0, 0.0);
  c10::complex<double> c(3.0, 0.0);
  c10::complex<double> z(0.0, 0.0);

  auto result = hypergeometric_2_f_1(a, b, c, z);
  EXPECT_NEAR(result.real(), 1.0, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(Hypergeometric2F1Test, ComplexDouble_SmallZ) {
  c10::complex<double> a(0.5, 0.0);
  c10::complex<double> b(0.5, 0.0);
  c10::complex<double> c(1.5, 0.0);
  c10::complex<double> z(0.25, 0.1);

  auto result = hypergeometric_2_f_1(a, b, c, z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

TEST_F(Hypergeometric2F1Test, ComplexDouble_RealAxisConsistency) {
  // On real axis, complex result should match real result
  double a = 0.5, b = 0.5, c = 1.5, z = 0.3;
  double real_result = hypergeometric_2_f_1(a, b, c, z);

  c10::complex<double> a_c(a, 0.0);
  c10::complex<double> b_c(b, 0.0);
  c10::complex<double> c_c(c, 0.0);
  c10::complex<double> z_c(z, 0.0);
  auto complex_result = hypergeometric_2_f_1(a_c, b_c, c_c, z_c);

  EXPECT_NEAR(complex_result.real(), real_result, 1e-12);
  EXPECT_NEAR(complex_result.imag(), 0.0, 1e-14);
}

// ============================================================================
// Symmetry Tests
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_ABSymmetry) {
  // 2F1(a, b; c; z) = 2F1(b, a; c; z)
  double a = 0.5, b = 1.5, c = 2.0, z = 0.3;
  double result1 = hypergeometric_2_f_1(a, b, c, z);
  double result2 = hypergeometric_2_f_1(b, a, c, z);
  EXPECT_NEAR(result1, result2, 1e-14);
}

// ============================================================================
// Convergence Tests
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_SmallZConvergence) {
  // Series should converge for |z| < 1
  std::vector<double> z_values = {0.1, 0.2, 0.3, 0.4, 0.5};
  for (double z : z_values) {
    double result = hypergeometric_2_f_1(0.5, 0.5, 1.5, z);
    EXPECT_FALSE(std::isnan(result)) << "Failed for z = " << z;
    EXPECT_FALSE(std::isinf(result)) << "Failed for z = " << z;
  }
}

TEST_F(Hypergeometric2F1Test, Double_NegativeZ) {
  // Should work for negative z
  double result = hypergeometric_2_f_1(0.5, 0.5, 1.5, -0.5);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_EqualAC) {
  // 2F1(a, b; a; z) = (1-z)^(-b)
  double a = 2.0, b = 1.5, z = 0.5;
  double expected = std::pow(1.0 - z, -b);
  double result = hypergeometric_2_f_1(a, b, a, z);
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(Hypergeometric2F1Test, Double_EqualBC) {
  // 2F1(a, b; b; z) = (1-z)^(-a)
  double a = 1.5, b = 2.0, z = 0.5;
  double expected = std::pow(1.0 - z, -a);
  double result = hypergeometric_2_f_1(a, b, b, z);
  EXPECT_NEAR(result, expected, 1e-10);
}

// ============================================================================
// Parameter Bounds Tests
// ============================================================================

TEST_F(Hypergeometric2F1Test, Double_SmallParameters) {
  double result = hypergeometric_2_f_1(0.1, 0.1, 0.5, 0.3);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

TEST_F(Hypergeometric2F1Test, Double_LargerParameters) {
  double result = hypergeometric_2_f_1(5.0, 3.0, 10.0, 0.3);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}
