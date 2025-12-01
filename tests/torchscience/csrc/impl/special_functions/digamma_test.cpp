#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/digamma.h"

using namespace torchscience::impl::special_functions;

class DigammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
  static constexpr double kEulerMascheroni = 0.5772156649015328606065;
};

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(DigammaTest, Float_One) {
  // psi(1) = -gamma (Euler-Mascheroni constant)
  float expected = -static_cast<float>(kEulerMascheroni);
  EXPECT_NEAR(digamma(1.0f), expected, 1e-5f);
}

TEST_F(DigammaTest, Float_PositiveIntegers) {
  // psi(n) = -gamma + sum_{k=1}^{n-1} 1/k for n >= 1
  float psi_2 = -static_cast<float>(kEulerMascheroni) + 1.0f;
  EXPECT_NEAR(digamma(2.0f), psi_2, 1e-5f);

  float psi_3 = psi_2 + 0.5f;
  EXPECT_NEAR(digamma(3.0f), psi_3, 1e-5f);

  float psi_4 = psi_3 + 1.0f / 3.0f;
  EXPECT_NEAR(digamma(4.0f), psi_4, 1e-5f);
}

TEST_F(DigammaTest, Float_HalfInteger) {
  // psi(1/2) = -gamma - 2*ln(2)
  float expected = -static_cast<float>(kEulerMascheroni) - 2.0f * std::log(2.0f);
  EXPECT_NEAR(digamma(0.5f), expected, 1e-5f);
}

TEST_F(DigammaTest, Float_Poles) {
  // Digamma has poles at non-positive integers
  EXPECT_TRUE(std::isnan(digamma(0.0f)));
  EXPECT_TRUE(std::isnan(digamma(-1.0f)));
  EXPECT_TRUE(std::isnan(digamma(-2.0f)));
  EXPECT_TRUE(std::isnan(digamma(-5.0f)));
}

TEST_F(DigammaTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(digamma(nan_val)));
}

TEST_F(DigammaTest, Float_NegativeNonInteger) {
  // Digamma exists for negative non-integers
  float x = -0.5f;
  float result = digamma(x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(DigammaTest, Double_One) {
  double expected = -kEulerMascheroni;
  EXPECT_NEAR(digamma(1.0), expected, 1e-11);
}

TEST_F(DigammaTest, Double_PositiveIntegers) {
  double psi_2 = -kEulerMascheroni + 1.0;
  EXPECT_NEAR(digamma(2.0), psi_2, 1e-11);

  double psi_3 = psi_2 + 0.5;
  EXPECT_NEAR(digamma(3.0), psi_3, 1e-11);

  double psi_4 = psi_3 + 1.0 / 3.0;
  EXPECT_NEAR(digamma(4.0), psi_4, 1e-11);

  double psi_5 = psi_4 + 0.25;
  EXPECT_NEAR(digamma(5.0), psi_5, 1e-11);
}

TEST_F(DigammaTest, Double_HalfInteger) {
  double expected = -kEulerMascheroni - 2.0 * std::log(2.0);
  EXPECT_NEAR(digamma(0.5), expected, 1e-11);
}

TEST_F(DigammaTest, Double_ThreeHalves) {
  // psi(3/2) = psi(1/2) + 2 = -gamma - 2*ln(2) + 2
  double expected = -kEulerMascheroni - 2.0 * std::log(2.0) + 2.0;
  EXPECT_NEAR(digamma(1.5), expected, 1e-11);
}

TEST_F(DigammaTest, Double_Poles) {
  EXPECT_TRUE(std::isnan(digamma(0.0)));
  EXPECT_TRUE(std::isnan(digamma(-1.0)));
  EXPECT_TRUE(std::isnan(digamma(-2.0)));
  EXPECT_TRUE(std::isnan(digamma(-10.0)));
}

TEST_F(DigammaTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(digamma(nan_val)));
}

TEST_F(DigammaTest, Double_LargePositive) {
  // For large x, psi(x) ~ ln(x) - 1/(2x)
  double x = 100.0;
  double result = digamma(x);
  double approx = std::log(x) - 1.0 / (2.0 * x);
  EXPECT_NEAR(result, approx, 0.01);
}

TEST_F(DigammaTest, Double_NegativeNonInteger) {
  double x = -0.5;
  double result = digamma(x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(DigammaTest, ComplexFloat_RealAxis) {
  c10::complex<float> z(2.0f, 0.0f);
  auto result = digamma(z);
  float expected = digamma(2.0f);
  EXPECT_NEAR(result.real(), expected, 1e-5f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(DigammaTest, ComplexFloat_Pole) {
  c10::complex<float> z(0.0f, 0.0f);
  auto result = digamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(DigammaTest, ComplexFloat_General) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = digamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(DigammaTest, ComplexDouble_RealAxis) {
  c10::complex<double> z(2.0, 0.0);
  auto result = digamma(z);
  double expected = digamma(2.0);
  EXPECT_NEAR(result.real(), expected, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(DigammaTest, ComplexDouble_Pole) {
  c10::complex<double> z(0.0, 0.0);
  auto result = digamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(DigammaTest, ComplexDouble_NegativeIntegerPole) {
  c10::complex<double> z(-1.0, 0.0);
  auto result = digamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(DigammaTest, ComplexDouble_General) {
  c10::complex<double> z(2.0, 1.0);
  auto result = digamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

TEST_F(DigammaTest, ComplexDouble_ConjugateSymmetry) {
  // psi(z*) = psi(z)*
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> z_conj(2.0, -1.0);

  auto result = digamma(z);
  auto result_conj = digamma(z_conj);

  EXPECT_NEAR(result.real(), result_conj.real(), 1e-14);
  EXPECT_NEAR(result.imag(), -result_conj.imag(), 1e-14);
}

// ============================================================================
// Recurrence Relation Tests
// ============================================================================

TEST_F(DigammaTest, Double_RecurrenceRelation) {
  // psi(x+1) = psi(x) + 1/x
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0};
  for (double x : test_values) {
    double lhs = digamma(x + 1.0);
    double rhs = digamma(x) + 1.0 / x;
    EXPECT_NEAR(lhs, rhs, 1e-13) << "Failed for x = " << x;
  }
}

TEST_F(DigammaTest, ComplexDouble_RecurrenceRelation) {
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> one(1.0, 0.0);

  auto lhs = digamma(z + one);
  auto rhs = digamma(z) + one / z;

  EXPECT_NEAR(lhs.real(), rhs.real(), 1e-12);
  EXPECT_NEAR(lhs.imag(), rhs.imag(), 1e-12);
}

// ============================================================================
// Reflection Formula Tests
// ============================================================================

TEST_F(DigammaTest, Double_ReflectionFormula) {
  // psi(1-x) - psi(x) = pi * cot(pi*x)
  std::vector<double> test_values = {0.1, 0.25, 0.3, 0.4, 0.7};
  for (double x : test_values) {
    double lhs = digamma(1.0 - x) - digamma(x);
    double rhs = kPi * std::cos(kPi * x) / std::sin(kPi * x);
    EXPECT_NEAR(lhs, rhs, 1e-10) << "Failed for x = " << x;
  }
}

// ============================================================================
// Asymptotic Behavior Tests
// ============================================================================

TEST_F(DigammaTest, Double_AsymptoticExpansion) {
  // For large x, psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + ...
  std::vector<double> test_values = {50.0, 100.0, 200.0};
  for (double x : test_values) {
    double result = digamma(x);
    double asymp = std::log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x);
    double rel_error = std::abs(result - asymp) / std::abs(result);
    EXPECT_LT(rel_error, 1e-6) << "Failed for x = " << x;
  }
}
