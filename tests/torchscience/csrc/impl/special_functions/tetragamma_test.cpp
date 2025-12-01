#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/tetragamma.h"
#include "impl/special_functions/trigamma.h"

using namespace torchscience::impl::special_functions;

class TetragammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
  // zeta(3) = Apery's constant
  static constexpr double kZeta3 = 1.2020569031595942853997381615114499907;
};

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(TetragammaTest, Float_One) {
  // psi''(1) = -2 * zeta(3)
  float expected = -2.0f * static_cast<float>(kZeta3);
  EXPECT_NEAR(tetragamma(1.0f), expected, 1e-3f);
}

TEST_F(TetragammaTest, Float_Poles) {
  EXPECT_TRUE(std::isnan(tetragamma(0.0f)));
  EXPECT_TRUE(std::isnan(tetragamma(-1.0f)));
  EXPECT_TRUE(std::isnan(tetragamma(-2.0f)));
}

TEST_F(TetragammaTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(tetragamma(nan_val)));
}

TEST_F(TetragammaTest, Float_Negative) {
  // Tetragamma is always negative for positive x
  std::vector<float> test_values = {0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
  for (float x : test_values) {
    EXPECT_LT(tetragamma(x), 0.0f) << "Failed for x = " << x;
  }
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(TetragammaTest, Double_One) {
  double expected = -2.0 * kZeta3;
  EXPECT_NEAR(tetragamma(1.0), expected, 1e-3);
}

TEST_F(TetragammaTest, Double_Two) {
  // psi''(2) = psi''(1) + 2/1^3 = -2*zeta(3) + 2
  double expected = -2.0 * kZeta3 + 2.0;
  EXPECT_NEAR(tetragamma(2.0), expected, 1e-3);
}

TEST_F(TetragammaTest, Double_Poles) {
  EXPECT_TRUE(std::isnan(tetragamma(0.0)));
  EXPECT_TRUE(std::isnan(tetragamma(-1.0)));
  EXPECT_TRUE(std::isnan(tetragamma(-2.0)));
  EXPECT_TRUE(std::isnan(tetragamma(-10.0)));
}

TEST_F(TetragammaTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(tetragamma(nan_val)));
}

TEST_F(TetragammaTest, Double_LargePositive) {
  // For large x, psi''(x) ~ -1/x^2 - 1/x^3 - ...
  double x = 100.0;
  double result = tetragamma(x);
  double approx = -1.0 / (x * x) - 1.0 / (x * x * x);
  EXPECT_NEAR(result, approx, 1e-6);
}

TEST_F(TetragammaTest, Double_NegativeNonInteger) {
  double x = -0.5;
  double result = tetragamma(x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(TetragammaTest, ComplexFloat_RealAxis) {
  c10::complex<float> z(1.0f, 0.0f);
  auto result = tetragamma(z);
  float expected = tetragamma(1.0f);
  EXPECT_NEAR(result.real(), expected, 1e-4f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(TetragammaTest, ComplexFloat_Pole) {
  c10::complex<float> z(0.0f, 0.0f);
  auto result = tetragamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(TetragammaTest, ComplexFloat_General) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = tetragamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(TetragammaTest, ComplexDouble_RealAxis) {
  c10::complex<double> z(1.0, 0.0);
  auto result = tetragamma(z);
  double expected = tetragamma(1.0);
  EXPECT_NEAR(result.real(), expected, 1e-10);
  EXPECT_NEAR(result.imag(), 0.0, 1e-10);
}

TEST_F(TetragammaTest, ComplexDouble_Pole) {
  c10::complex<double> z(0.0, 0.0);
  auto result = tetragamma(z);
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(TetragammaTest, ComplexDouble_General) {
  c10::complex<double> z(2.0, 1.0);
  auto result = tetragamma(z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

TEST_F(TetragammaTest, ComplexDouble_ConjugateSymmetry) {
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> z_conj(2.0, -1.0);

  auto result = tetragamma(z);
  auto result_conj = tetragamma(z_conj);

  EXPECT_NEAR(result.real(), result_conj.real(), 1e-10);
  EXPECT_NEAR(result.imag(), -result_conj.imag(), 1e-10);
}

// ============================================================================
// Recurrence Relation Tests
// ============================================================================

TEST_F(TetragammaTest, Double_RecurrenceRelation) {
  // psi''(x+1) = psi''(x) + 2/x^3
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0};
  for (double x : test_values) {
    double lhs = tetragamma(x + 1.0);
    double rhs = tetragamma(x) + 2.0 / (x * x * x);
    EXPECT_NEAR(lhs, rhs, 1e-3) << "Failed for x = " << x;
  }
}

TEST_F(TetragammaTest, ComplexDouble_RecurrenceRelation) {
  c10::complex<double> z(2.0, 1.0);
  c10::complex<double> one(1.0, 0.0);
  c10::complex<double> two(2.0, 0.0);

  auto lhs = tetragamma(z + one);
  auto rhs = tetragamma(z) + two / (z * z * z);

  EXPECT_NEAR(lhs.real(), rhs.real(), 1e-10);
  EXPECT_NEAR(lhs.imag(), rhs.imag(), 1e-10);
}

// ============================================================================
// Finite Difference Derivative Tests
// ============================================================================

TEST_F(TetragammaTest, Double_DerivativeOfTrigamma) {
  // Tetragamma is the derivative of trigamma
  std::vector<double> test_values = {1.0, 2.0, 3.0, 5.0, 10.0};
  for (double x : test_values) {
    double eps = 1e-6;
    double numerical_deriv = (trigamma(x + eps) - trigamma(x - eps)) / (2.0 * eps);
    double analytical = tetragamma(x);
    EXPECT_NEAR(analytical, numerical_deriv, 1e-3)
        << "Failed for x = " << x;
  }
}

// ============================================================================
// Reflection Formula Tests
// ============================================================================

TEST_F(TetragammaTest, Double_ReflectionFormula) {
  // psi''(1-x) - psi''(x) = 2*pi^3 * cos(pi*x) / sin^3(pi*x)
  std::vector<double> test_values = {0.1, 0.25, 0.3, 0.4};
  for (double x : test_values) {
    double lhs = tetragamma(1.0 - x) - tetragamma(x);
    double sin_pi_x = std::sin(kPi * x);
    double cos_pi_x = std::cos(kPi * x);
    double rhs = 2.0 * kPi * kPi * kPi * cos_pi_x / (sin_pi_x * sin_pi_x * sin_pi_x);
    EXPECT_NEAR(lhs, rhs, std::abs(rhs) * 1e-3) << "Failed for x = " << x;
  }
}
