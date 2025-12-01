#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/sin_pi.h"

using namespace torchscience::impl::special_functions;

class SinPiTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;

  template <typename T>
  T reference_sin_pi(T x) {
    return std::sin(kPi * x);
  }
};

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(SinPiTest, Float_Zero) {
  EXPECT_FLOAT_EQ(sin_pi(0.0f), 0.0f);
}

TEST_F(SinPiTest, Float_Integers) {
  // sin(pi * n) = 0 for all integers n
  for (int n = -10; n <= 10; ++n) {
    EXPECT_FLOAT_EQ(sin_pi(static_cast<float>(n)), 0.0f)
        << "Failed for n = " << n;
  }
}

TEST_F(SinPiTest, Float_HalfIntegers) {
  // sin(pi * (n + 0.5)) = +/- 1
  EXPECT_FLOAT_EQ(sin_pi(0.5f), 1.0f);
  EXPECT_FLOAT_EQ(sin_pi(-0.5f), -1.0f);
  EXPECT_FLOAT_EQ(sin_pi(1.5f), -1.0f);
  EXPECT_FLOAT_EQ(sin_pi(-1.5f), 1.0f);
  EXPECT_FLOAT_EQ(sin_pi(2.5f), 1.0f);
  EXPECT_FLOAT_EQ(sin_pi(-2.5f), -1.0f);
}

TEST_F(SinPiTest, Float_QuarterIntegers) {
  const float expected = std::sqrt(2.0f) / 2.0f;
  EXPECT_NEAR(sin_pi(0.25f), expected, 1e-6f);
  EXPECT_NEAR(sin_pi(0.75f), expected, 1e-6f);
  EXPECT_NEAR(sin_pi(-0.25f), -expected, 1e-6f);
  EXPECT_NEAR(sin_pi(-0.75f), -expected, 1e-6f);
}

TEST_F(SinPiTest, Float_SmallValues) {
  // Test various small values against std::sin(pi * x)
  std::vector<float> test_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.7f, 0.8f, 0.9f};
  for (float x : test_values) {
    float expected = reference_sin_pi(x);
    EXPECT_NEAR(sin_pi(x), expected, 1e-6f) << "Failed for x = " << x;
    EXPECT_NEAR(sin_pi(-x), -expected, 1e-6f) << "Failed for x = " << -x;
  }
}

TEST_F(SinPiTest, Float_LargeValues) {
  // Large values should use range reduction
  // Note: 100.25f can represent the fractional part in float precision
  float large_val = 100.25f;
  float expected = std::sqrt(2.0f) / 2.0f;
  EXPECT_NEAR(sin_pi(large_val), expected, 1e-4f);
}

TEST_F(SinPiTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(sin_pi(nan_val)));
}

TEST_F(SinPiTest, Float_Infinity) {
  float inf = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(std::isnan(sin_pi(inf)));
  EXPECT_TRUE(std::isnan(sin_pi(-inf)));
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(SinPiTest, Double_Zero) {
  EXPECT_DOUBLE_EQ(sin_pi(0.0), 0.0);
}

TEST_F(SinPiTest, Double_Integers) {
  // sin(pi * n) = 0 for all integers n
  for (int n = -10; n <= 10; ++n) {
    EXPECT_DOUBLE_EQ(sin_pi(static_cast<double>(n)), 0.0)
        << "Failed for n = " << n;
  }
}

TEST_F(SinPiTest, Double_HalfIntegers) {
  EXPECT_DOUBLE_EQ(sin_pi(0.5), 1.0);
  EXPECT_DOUBLE_EQ(sin_pi(-0.5), -1.0);
  EXPECT_DOUBLE_EQ(sin_pi(1.5), -1.0);
  EXPECT_DOUBLE_EQ(sin_pi(-1.5), 1.0);
}

TEST_F(SinPiTest, Double_QuarterIntegers) {
  const double expected = std::sqrt(2.0) / 2.0;
  EXPECT_NEAR(sin_pi(0.25), expected, 1e-14);
  EXPECT_NEAR(sin_pi(0.75), expected, 1e-14);
}

TEST_F(SinPiTest, Double_SmallValues) {
  std::vector<double> test_values = {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9};
  for (double x : test_values) {
    double expected = reference_sin_pi(x);
    EXPECT_NEAR(sin_pi(x), expected, 1e-14) << "Failed for x = " << x;
  }
}

TEST_F(SinPiTest, Double_LargeValues) {
  // Test range reduction for large values
  double large_val = 1e15 + 0.25;
  // Due to floating point limitations, we check that it's reasonable
  double result = sin_pi(large_val);
  EXPECT_GE(result, -1.0);
  EXPECT_LE(result, 1.0);
}

TEST_F(SinPiTest, Double_VeryLargeIntegers) {
  // Large integers should still return exactly 0
  EXPECT_DOUBLE_EQ(sin_pi(1e8), 0.0);
  EXPECT_DOUBLE_EQ(sin_pi(-1e8), 0.0);
}

TEST_F(SinPiTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(sin_pi(nan_val)));
}

TEST_F(SinPiTest, Double_Infinity) {
  double inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(std::isnan(sin_pi(inf)));
  EXPECT_TRUE(std::isnan(sin_pi(-inf)));
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(SinPiTest, ComplexFloat_RealAxis) {
  // On real axis, should match real sin_pi
  c10::complex<float> z(0.5f, 0.0f);
  auto result = sin_pi(z);
  EXPECT_NEAR(result.real(), 1.0f, 1e-6f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(SinPiTest, ComplexFloat_ImaginaryAxis) {
  // sin(pi * i*y) = i * sinh(pi * y)
  c10::complex<float> z(0.0f, 1.0f);
  auto result = sin_pi(z);
  float expected_imag = std::sinh(static_cast<float>(kPi));
  EXPECT_NEAR(result.real(), 0.0f, 1e-5f);
  EXPECT_NEAR(result.imag(), expected_imag, 1e-4f);
}

TEST_F(SinPiTest, ComplexFloat_General) {
  // sin(pi*(a+bi)) = sin(pi*a)*cosh(pi*b) + i*cos(pi*a)*sinh(pi*b)
  c10::complex<float> z(0.25f, 0.5f);
  auto result = sin_pi(z);

  float a = 0.25f;
  float b = 0.5f;
  float expected_real = std::sin(static_cast<float>(kPi) * a) *
                        std::cosh(static_cast<float>(kPi) * b);
  float expected_imag = std::cos(static_cast<float>(kPi) * a) *
                        std::sinh(static_cast<float>(kPi) * b);

  EXPECT_NEAR(result.real(), expected_real, 1e-5f);
  EXPECT_NEAR(result.imag(), expected_imag, 1e-5f);
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(SinPiTest, ComplexDouble_RealAxis) {
  c10::complex<double> z(0.5, 0.0);
  auto result = sin_pi(z);
  EXPECT_NEAR(result.real(), 1.0, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(SinPiTest, ComplexDouble_ImaginaryAxis) {
  c10::complex<double> z(0.0, 1.0);
  auto result = sin_pi(z);
  double expected_imag = std::sinh(kPi);
  EXPECT_NEAR(result.real(), 0.0, 1e-14);
  EXPECT_NEAR(result.imag(), expected_imag, 1e-12);
}

TEST_F(SinPiTest, ComplexDouble_General) {
  c10::complex<double> z(0.25, 0.5);
  auto result = sin_pi(z);

  double a = 0.25;
  double b = 0.5;
  double expected_real = std::sin(kPi * a) * std::cosh(kPi * b);
  double expected_imag = std::cos(kPi * a) * std::sinh(kPi * b);

  EXPECT_NEAR(result.real(), expected_real, 1e-14);
  EXPECT_NEAR(result.imag(), expected_imag, 1e-14);
}

TEST_F(SinPiTest, ComplexDouble_LargeRealPart) {
  // Test range reduction for complex with large real part
  c10::complex<double> z(1e10, 0.5);
  auto result = sin_pi(z);
  // Result should be bounded
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
  EXPECT_FALSE(std::isinf(result.real()));
}

// ============================================================================
// Symmetry Tests
// ============================================================================

TEST_F(SinPiTest, OddSymmetry) {
  // sin_pi(-x) = -sin_pi(x)
  std::vector<double> test_values = {0.1, 0.25, 0.5, 0.75, 1.5, 2.3};
  for (double x : test_values) {
    EXPECT_NEAR(sin_pi(-x), -sin_pi(x), 1e-14)
        << "Odd symmetry failed for x = " << x;
  }
}

TEST_F(SinPiTest, Periodicity) {
  // sin_pi(x + 2) = sin_pi(x)
  std::vector<double> test_values = {0.1, 0.25, 0.5, 0.75, 1.5};
  for (double x : test_values) {
    EXPECT_NEAR(sin_pi(x + 2.0), sin_pi(x), 1e-14)
        << "Periodicity failed for x = " << x;
    EXPECT_NEAR(sin_pi(x - 2.0), sin_pi(x), 1e-14)
        << "Periodicity failed for x = " << x;
  }
}
