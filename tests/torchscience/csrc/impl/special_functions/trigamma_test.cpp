#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/digamma.h"
#include "impl/special_functions/trigamma.h"

using namespace torchscience::impl::special_functions;

class TrigammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(TrigammaTest, Float_One) {
  // psi'(1) = pi^2 / 6
  float expected = static_cast<float>(kPi * kPi / 6.0);
  EXPECT_NEAR(trigamma(1.0f), expected, 1e-5f);
}

TEST_F(TrigammaTest, Float_PositiveIntegers) {
  // psi'(n) = pi^2/6 - sum_{k=1}^{n-1} 1/k^2
  float psi1_1 = static_cast<float>(kPi * kPi / 6.0);
  EXPECT_NEAR(trigamma(1.0f), psi1_1, 1e-5f);

  float psi1_2 = psi1_1 - 1.0f;
  EXPECT_NEAR(trigamma(2.0f), psi1_2, 1e-5f);

  float psi1_3 = psi1_2 - 0.25f;
  EXPECT_NEAR(trigamma(3.0f), psi1_3, 1e-5f);
}

TEST_F(TrigammaTest, Float_Half) {
  // psi'(1/2) = pi^2 / 2
  float expected = static_cast<float>(kPi * kPi / 2.0);
  EXPECT_NEAR(trigamma(0.5f), expected, 1e-4f);
}

TEST_F(TrigammaTest, Float_Poles) {
  EXPECT_TRUE(std::isnan(trigamma(0.0f)));
  EXPECT_TRUE(std::isnan(trigamma(-1.0f)));
  EXPECT_TRUE(std::isnan(trigamma(-2.0f)));
}

TEST_F(TrigammaTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(trigamma(nan_val)));
}

TEST_F(TrigammaTest, Float_Positive) {
  // Trigamma is always positive for positive x
  std::vector<float> test_values = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
  for (float x : test_values) {
    EXPECT_GT(trigamma(x), 0.0f) << "Failed for x = " << x;
  }
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(TrigammaTest, Double_One) {
  double expected = kPi * kPi / 6.0;
  EXPECT_NEAR(trigamma(1.0), expected, 1e-10);
}

TEST_F(TrigammaTest, Double_PositiveIntegers) {
  double psi1_1 = kPi * kPi / 6.0;
  EXPECT_NEAR(trigamma(1.0), psi1_1, 1e-10);

  double psi1_2 = psi1_1 - 1.0;
  EXPECT_NEAR(trigamma(2.0), psi1_2, 1e-10);

  double psi1_3 = psi1_2 - 0.25;
  EXPECT_NEAR(trigamma(3.0), psi1_3, 1e-10);

  double psi1_4 = psi1_3 - 1.0 / 9.0;
  EXPECT_NEAR(trigamma(4.0), psi1_4, 1e-10);
}

TEST_F(TrigammaTest, Double_Half) {
  double expected = kPi * kPi / 2.0;
  EXPECT_NEAR(trigamma(0.5), expected, 1e-10);
}

TEST_F(TrigammaTest, Double_Poles) {
  EXPECT_TRUE(std::isnan(trigamma(0.0)));
  EXPECT_TRUE(std::isnan(trigamma(-1.0)));
  EXPECT_TRUE(std::isnan(trigamma(-2.0)));
  EXPECT_TRUE(std::isnan(trigamma(-10.0)));
}

TEST_F(TrigammaTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(trigamma(nan_val)));
}

TEST_F(TrigammaTest, Double_LargePositive) {
  // For large x, psi'(x) ~ 1/x + 1/(2x^2) + ...
  double x = 100.0;
  double result = trigamma(x);
  double approx = 1.0 / x + 1.0 / (2.0 * x * x);
  EXPECT_NEAR(result, approx, 1e-6);
}

TEST_F(TrigammaTest, Double_NegativeNonInteger) {
  double x = -0.5;
  double result = trigamma(x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(TrigammaTest, ComplexFloat_RealAxis) {
  c10::complex<float> z(1.0f, 0.0f);
  auto result = trigamma(z);
  float expected = trigamma(1.0f);
  EXPECT_NEAR(result.real(), expected, 1e-5f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(TrigammaTest, ComplexFloat_Pole) {
  c10::complex<float> z(0.0f, 0.0f);
  auto result = trigamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(TrigammaTest, ComplexFloat_General) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = trigamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(TrigammaTest, ComplexDouble_RealAxis) {
  c10::complex<double> z(1.0, 0.0);
  auto result = trigamma(z);
  double expected = trigamma(1.0);
  EXPECT_NEAR(result.real(), expected, 1e-10);
  EXPECT_NEAR(result.imag(), 0.0, 1e-10);
}

TEST_F(TrigammaTest, ComplexDouble_Pole) {
  c10::complex<double> z(0.0, 0.0);
  auto result = trigamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(TrigammaTest, ComplexDouble_General) {
  c10::complex<double> z(2.0, 1.0);
  auto result = trigamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

TEST_F(TrigammaTest, ComplexDouble_ConjugateSymmetry) {
  // psi'(z*) = psi'(z)*
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> z_conj(2.0, -1.0);

  auto result = trigamma(z);
  auto result_conj = trigamma(z_conj);

  EXPECT_NEAR(result.real(), result_conj.real(), 1e-10);
  EXPECT_NEAR(result.imag(), -result_conj.imag(), 1e-10);
}

// ============================================================================
// Recurrence Relation Tests
// ============================================================================

TEST_F(TrigammaTest, Double_RecurrenceRelation) {
  // psi'(x+1) = psi'(x) - 1/x^2
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0};
  for (double x : test_values) {
    double lhs = trigamma(x + 1.0);
    double rhs = trigamma(x) - 1.0 / (x * x);
    EXPECT_NEAR(lhs, rhs, 1e-10) << "Failed for x = " << x;
  }
}

TEST_F(TrigammaTest, ComplexDouble_RecurrenceRelation) {
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> one(1.0, 0.0);

  auto lhs = trigamma(z + one);
  auto rhs = trigamma(z) - one / (z * z);

  EXPECT_NEAR(lhs.real(), rhs.real(), 1e-12);
  EXPECT_NEAR(lhs.imag(), rhs.imag(), 1e-12);
}

// ============================================================================
// Reflection Formula Tests
// ============================================================================

TEST_F(TrigammaTest, Double_ReflectionFormula) {
  // psi'(1-x) + psi'(x) = pi^2 / sin^2(pi*x)
  std::vector<double> test_values = {0.1, 0.25, 0.3, 0.4, 0.7};
  for (double x : test_values) {
    double lhs = trigamma(1.0 - x) + trigamma(x);
    double sin_pi_x = std::sin(kPi * x);
    double rhs = kPi * kPi / (sin_pi_x * sin_pi_x);
    EXPECT_NEAR(lhs, rhs, 1e-10) << "Failed for x = " << x;
  }
}

// ============================================================================
// Finite Difference Derivative Tests
// ============================================================================

TEST_F(TrigammaTest, Double_DerivativeOfDigamma) {
  // Trigamma is the derivative of digamma
  std::vector<double> test_values = {1.0, 2.0, 3.0, 5.0, 10.0};
  for (double x : test_values) {
    double eps = 1e-6;
    double numerical_deriv = (digamma(x + eps) - digamma(x - eps)) / (2.0 * eps);
    double analytical = trigamma(x);
    EXPECT_NEAR(analytical, numerical_deriv, 1e-5)
        << "Failed for x = " << x;
  }
}
