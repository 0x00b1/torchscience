#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/gamma.h"

using namespace torchscience::impl::special_functions;

class GammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Float Tests - Positive Integers
// ============================================================================

TEST_F(GammaTest, Float_PositiveIntegers) {
  // Gamma(n) = (n-1)!
  EXPECT_FLOAT_EQ(gamma(1.0f), 1.0f);   // 0!
  EXPECT_FLOAT_EQ(gamma(2.0f), 1.0f);   // 1!
  EXPECT_FLOAT_EQ(gamma(3.0f), 2.0f);   // 2!
  EXPECT_FLOAT_EQ(gamma(4.0f), 6.0f);   // 3!
  EXPECT_FLOAT_EQ(gamma(5.0f), 24.0f);  // 4!
  EXPECT_FLOAT_EQ(gamma(6.0f), 120.0f); // 5!
  EXPECT_FLOAT_EQ(gamma(7.0f), 720.0f); // 6!
}

TEST_F(GammaTest, Float_HalfInteger) {
  // Gamma(1/2) = sqrt(pi)
  float expected = std::sqrt(static_cast<float>(kPi));
  EXPECT_NEAR(gamma(0.5f), expected, 1e-5f);
}

TEST_F(GammaTest, Float_ThreeHalves) {
  // Gamma(3/2) = sqrt(pi)/2
  float expected = std::sqrt(static_cast<float>(kPi)) / 2.0f;
  EXPECT_NEAR(gamma(1.5f), expected, 1e-5f);
}

TEST_F(GammaTest, Float_Poles) {
  // Gamma has poles at non-positive integers
  EXPECT_TRUE(std::isinf(gamma(0.0f)));
  EXPECT_TRUE(std::isinf(gamma(-1.0f)));
  EXPECT_TRUE(std::isinf(gamma(-2.0f)));
  EXPECT_TRUE(std::isinf(gamma(-5.0f)));
}

TEST_F(GammaTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(gamma(nan_val)));
}

TEST_F(GammaTest, Float_NegativeNonInteger) {
  // Gamma exists for negative non-integers
  float result = gamma(-0.5f);
  // Gamma(-1/2) = -2*sqrt(pi)
  float expected = -2.0f * std::sqrt(static_cast<float>(kPi));
  EXPECT_NEAR(result, expected, 1e-4f);
}

// ============================================================================
// Double Tests - Positive Integers
// ============================================================================

TEST_F(GammaTest, Double_PositiveIntegers) {
  EXPECT_DOUBLE_EQ(gamma(1.0), 1.0);
  EXPECT_DOUBLE_EQ(gamma(2.0), 1.0);
  EXPECT_DOUBLE_EQ(gamma(3.0), 2.0);
  EXPECT_DOUBLE_EQ(gamma(4.0), 6.0);
  EXPECT_DOUBLE_EQ(gamma(5.0), 24.0);
  EXPECT_DOUBLE_EQ(gamma(6.0), 120.0);
  EXPECT_DOUBLE_EQ(gamma(7.0), 720.0);
  EXPECT_DOUBLE_EQ(gamma(8.0), 5040.0);
  EXPECT_DOUBLE_EQ(gamma(9.0), 40320.0);
  EXPECT_DOUBLE_EQ(gamma(10.0), 362880.0);
  EXPECT_DOUBLE_EQ(gamma(11.0), 3628800.0);
}

TEST_F(GammaTest, Double_HalfInteger) {
  double expected = std::sqrt(kPi);
  EXPECT_NEAR(gamma(0.5), expected, 1e-14);
}

TEST_F(GammaTest, Double_ThreeHalves) {
  double expected = std::sqrt(kPi) / 2.0;
  EXPECT_NEAR(gamma(1.5), expected, 1e-14);
}

TEST_F(GammaTest, Double_FiveHalves) {
  // Gamma(5/2) = (3/2) * (1/2) * sqrt(pi) = (3/4) * sqrt(pi)
  double expected = 0.75 * std::sqrt(kPi);
  EXPECT_NEAR(gamma(2.5), expected, 1e-14);
}

TEST_F(GammaTest, Double_Poles) {
  EXPECT_TRUE(std::isinf(gamma(0.0)));
  EXPECT_TRUE(std::isinf(gamma(-1.0)));
  EXPECT_TRUE(std::isinf(gamma(-2.0)));
  EXPECT_TRUE(std::isinf(gamma(-10.0)));
}

TEST_F(GammaTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(gamma(nan_val)));
}

TEST_F(GammaTest, Double_NegativeNonInteger) {
  // Gamma(-1/2) = -2*sqrt(pi)
  double expected = -2.0 * std::sqrt(kPi);
  EXPECT_NEAR(gamma(-0.5), expected, 1e-13);
}

TEST_F(GammaTest, Double_NegativeThreeHalves) {
  // Gamma(-3/2) = (4/3)*sqrt(pi)
  double expected = (4.0 / 3.0) * std::sqrt(kPi);
  EXPECT_NEAR(gamma(-1.5), expected, 1e-13);
}

// ============================================================================
// Double Tests - Large Values
// ============================================================================

TEST_F(GammaTest, Double_LargePositive) {
  // Gamma(171) should be finite but near overflow
  double result = gamma(171.0);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_GT(result, 0.0);

  // Gamma(172) should overflow
  double result_overflow = gamma(172.0);
  EXPECT_TRUE(std::isinf(result_overflow));
}

TEST_F(GammaTest, Double_VeryLargeNegative) {
  // For very large negative z, Gamma(z) should approach 0
  double result = gamma(-100.5);
  EXPECT_FALSE(std::isinf(result));
  EXPECT_FALSE(std::isnan(result));
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(GammaTest, ComplexFloat_RealAxis) {
  c10::complex<float> z(3.0f, 0.0f);
  auto result = gamma(z);
  EXPECT_NEAR(result.real(), 2.0f, 1e-5f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(GammaTest, ComplexFloat_HalfInteger) {
  c10::complex<float> z(0.5f, 0.0f);
  auto result = gamma(z);
  float expected = std::sqrt(static_cast<float>(kPi));
  EXPECT_NEAR(result.real(), expected, 1e-5f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(GammaTest, ComplexFloat_Pole) {
  c10::complex<float> z(0.0f, 0.0f);
  auto result = gamma(z);
  EXPECT_TRUE(std::isinf(result.real()));
}

TEST_F(GammaTest, ComplexFloat_General) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = gamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
  EXPECT_FALSE(std::isinf(result.real()));
  EXPECT_FALSE(std::isinf(result.imag()));
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(GammaTest, ComplexDouble_RealAxis) {
  c10::complex<double> z(3.0, 0.0);
  auto result = gamma(z);
  EXPECT_NEAR(result.real(), 2.0, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(GammaTest, ComplexDouble_HalfInteger) {
  c10::complex<double> z(0.5, 0.0);
  auto result = gamma(z);
  double expected = std::sqrt(kPi);
  EXPECT_NEAR(result.real(), expected, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(GammaTest, ComplexDouble_Pole) {
  c10::complex<double> z(0.0, 0.0);
  auto result = gamma(z);
  EXPECT_TRUE(std::isinf(result.real()));
}

TEST_F(GammaTest, ComplexDouble_ImaginaryAxis) {
  c10::complex<double> z(0.5, 1.0);
  auto result = gamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

TEST_F(GammaTest, ComplexDouble_ConjugateSymmetry) {
  // Gamma(z*) = Gamma(z)*
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> z_conj(2.0, -1.0);

  auto result = gamma(z);
  auto result_conj = gamma(z_conj);

  EXPECT_NEAR(result.real(), result_conj.real(), 1e-12);
  EXPECT_NEAR(result.imag(), -result_conj.imag(), 1e-12);
}

// ============================================================================
// Functional Equation Tests
// ============================================================================

TEST_F(GammaTest, Double_RecurrenceRelation) {
  // Gamma(z+1) = z * Gamma(z)
  std::vector<double> test_values = {0.5, 1.5, 2.5, 3.7, 5.2};
  for (double z : test_values) {
    double lhs = gamma(z + 1.0);
    double rhs = z * gamma(z);
    EXPECT_NEAR(lhs, rhs, std::abs(lhs) * 1e-13)
        << "Failed for z = " << z;
  }
}

TEST_F(GammaTest, Double_ReflectionFormula) {
  // Gamma(z) * Gamma(1-z) = pi / sin(pi*z)
  std::vector<double> test_values = {0.1, 0.25, 0.3, 0.4, 0.75};
  for (double z : test_values) {
    double lhs = gamma(z) * gamma(1.0 - z);
    double rhs = kPi / std::sin(kPi * z);
    EXPECT_NEAR(lhs, rhs, std::abs(lhs) * 1e-12)
        << "Failed for z = " << z;
  }
}

TEST_F(GammaTest, ComplexDouble_RecurrenceRelation) {
  c10::complex<double> z(2.0, 1.0);
  auto lhs = gamma(z + c10::complex<double>(1.0, 0.0));
  auto rhs = z * gamma(z);
  EXPECT_NEAR(lhs.real(), rhs.real(), std::abs(lhs.real()) * 1e-12);
  EXPECT_NEAR(lhs.imag(), rhs.imag(), std::abs(lhs.imag()) * 1e-12);
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST_F(GammaTest, Double_Backward) {
  double z = 2.5;
  double grad_output = 1.0;
  double grad_z = gamma_backward(grad_output, z);

  // d/dz Gamma(z) = Gamma(z) * psi(z)
  // Finite difference check
  double eps = 1e-7;
  double numerical_grad = (gamma(z + eps) - gamma(z - eps)) / (2.0 * eps);
  EXPECT_NEAR(grad_z, numerical_grad, 1e-6);
}

TEST_F(GammaTest, Float_Backward) {
  float z = 2.5f;
  float grad_output = 1.0f;
  float grad_z = gamma_backward(grad_output, z);

  float eps = 1e-4f;
  float numerical_grad = (gamma(z + eps) - gamma(z - eps)) / (2.0f * eps);
  EXPECT_NEAR(grad_z, numerical_grad, 1e-2f);
}
