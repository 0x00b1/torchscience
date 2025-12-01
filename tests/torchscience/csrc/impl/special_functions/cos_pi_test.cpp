#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/cos_pi.h"

using namespace torchscience::impl::special_functions;

class CosPiTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;

  template <typename T>
  T reference_cos_pi(T x) {
    return std::cos(kPi * x);
  }
};

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(CosPiTest, Float_Zero) {
  EXPECT_FLOAT_EQ(cos_pi(0.0f), 1.0f);
}

TEST_F(CosPiTest, Float_Integers) {
  // cos(pi * n) = (-1)^n for all integers n
  for (int n = -10; n <= 10; ++n) {
    float expected = (n % 2 == 0) ? 1.0f : -1.0f;
    EXPECT_FLOAT_EQ(cos_pi(static_cast<float>(n)), expected)
        << "Failed for n = " << n;
  }
}

TEST_F(CosPiTest, Float_HalfIntegers) {
  // cos(pi * (n + 0.5)) = 0
  EXPECT_FLOAT_EQ(cos_pi(0.5f), 0.0f);
  EXPECT_FLOAT_EQ(cos_pi(-0.5f), 0.0f);
  EXPECT_FLOAT_EQ(cos_pi(1.5f), 0.0f);
  EXPECT_FLOAT_EQ(cos_pi(-1.5f), 0.0f);
  EXPECT_FLOAT_EQ(cos_pi(2.5f), 0.0f);
  EXPECT_FLOAT_EQ(cos_pi(-2.5f), 0.0f);
}

TEST_F(CosPiTest, Float_QuarterIntegers) {
  const float expected = std::sqrt(2.0f) / 2.0f;
  EXPECT_NEAR(cos_pi(0.25f), expected, 1e-6f);
  EXPECT_NEAR(cos_pi(-0.25f), expected, 1e-6f);
  EXPECT_NEAR(cos_pi(0.75f), -expected, 1e-6f);
  EXPECT_NEAR(cos_pi(-0.75f), -expected, 1e-6f);
}

TEST_F(CosPiTest, Float_SmallValues) {
  std::vector<float> test_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.7f, 0.8f, 0.9f};
  for (float x : test_values) {
    float expected = reference_cos_pi(x);
    EXPECT_NEAR(cos_pi(x), expected, 1e-6f) << "Failed for x = " << x;
    EXPECT_NEAR(cos_pi(-x), expected, 1e-6f) << "Failed for x = " << -x;
  }
}

TEST_F(CosPiTest, Float_LargeValues) {
  // Large values should use range reduction
  float large_val = 1e10f;
  // 1e10 is an even number, so cos_pi(1e10) = 1
  EXPECT_NEAR(cos_pi(large_val), 1.0f, 1e-4f);
}

TEST_F(CosPiTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(cos_pi(nan_val)));
}

TEST_F(CosPiTest, Float_Infinity) {
  float inf = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(std::isnan(cos_pi(inf)));
  EXPECT_TRUE(std::isnan(cos_pi(-inf)));
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(CosPiTest, Double_Zero) {
  EXPECT_DOUBLE_EQ(cos_pi(0.0), 1.0);
}

TEST_F(CosPiTest, Double_Integers) {
  for (int n = -10; n <= 10; ++n) {
    double expected = (n % 2 == 0) ? 1.0 : -1.0;
    EXPECT_DOUBLE_EQ(cos_pi(static_cast<double>(n)), expected)
        << "Failed for n = " << n;
  }
}

TEST_F(CosPiTest, Double_HalfIntegers) {
  EXPECT_DOUBLE_EQ(cos_pi(0.5), 0.0);
  EXPECT_DOUBLE_EQ(cos_pi(-0.5), 0.0);
  EXPECT_DOUBLE_EQ(cos_pi(1.5), 0.0);
  EXPECT_DOUBLE_EQ(cos_pi(-1.5), 0.0);
}

TEST_F(CosPiTest, Double_QuarterIntegers) {
  const double expected = std::sqrt(2.0) / 2.0;
  EXPECT_NEAR(cos_pi(0.25), expected, 1e-14);
  EXPECT_NEAR(cos_pi(-0.25), expected, 1e-14);
  EXPECT_NEAR(cos_pi(0.75), -expected, 1e-14);
}

TEST_F(CosPiTest, Double_SmallValues) {
  std::vector<double> test_values = {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9};
  for (double x : test_values) {
    double expected = reference_cos_pi(x);
    EXPECT_NEAR(cos_pi(x), expected, 1e-14) << "Failed for x = " << x;
  }
}

TEST_F(CosPiTest, Double_LargeValues) {
  // Test range reduction for large values
  double large_val = 1e15;
  // Result should be bounded
  double result = cos_pi(large_val);
  EXPECT_GE(result, -1.0);
  EXPECT_LE(result, 1.0);
}

TEST_F(CosPiTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(cos_pi(nan_val)));
}

TEST_F(CosPiTest, Double_Infinity) {
  double inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(std::isnan(cos_pi(inf)));
  EXPECT_TRUE(std::isnan(cos_pi(-inf)));
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(CosPiTest, ComplexFloat_RealAxis) {
  c10::complex<float> z(0.0f, 0.0f);
  auto result = cos_pi(z);
  EXPECT_NEAR(result.real(), 1.0f, 1e-6f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(CosPiTest, ComplexFloat_ImaginaryAxis) {
  // cos(pi * i*y) = cosh(pi * y)
  c10::complex<float> z(0.0f, 1.0f);
  auto result = cos_pi(z);
  float expected_real = std::cosh(static_cast<float>(kPi));
  EXPECT_NEAR(result.real(), expected_real, 1e-4f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(CosPiTest, ComplexFloat_General) {
  // cos(pi*(a+bi)) = cos(pi*a)*cosh(pi*b) - i*sin(pi*a)*sinh(pi*b)
  c10::complex<float> z(0.25f, 0.5f);
  auto result = cos_pi(z);

  float a = 0.25f;
  float b = 0.5f;
  float expected_real = std::cos(static_cast<float>(kPi) * a) *
                        std::cosh(static_cast<float>(kPi) * b);
  float expected_imag = -std::sin(static_cast<float>(kPi) * a) *
                        std::sinh(static_cast<float>(kPi) * b);

  EXPECT_NEAR(result.real(), expected_real, 1e-5f);
  EXPECT_NEAR(result.imag(), expected_imag, 1e-5f);
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(CosPiTest, ComplexDouble_RealAxis) {
  c10::complex<double> z(0.0, 0.0);
  auto result = cos_pi(z);
  EXPECT_NEAR(result.real(), 1.0, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(CosPiTest, ComplexDouble_ImaginaryAxis) {
  c10::complex<double> z(0.0, 1.0);
  auto result = cos_pi(z);
  double expected_real = std::cosh(kPi);
  EXPECT_NEAR(result.real(), expected_real, 1e-12);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(CosPiTest, ComplexDouble_General) {
  c10::complex<double> z(0.25, 0.5);
  auto result = cos_pi(z);

  double a = 0.25;
  double b = 0.5;
  double expected_real = std::cos(kPi * a) * std::cosh(kPi * b);
  double expected_imag = -std::sin(kPi * a) * std::sinh(kPi * b);

  EXPECT_NEAR(result.real(), expected_real, 1e-14);
  EXPECT_NEAR(result.imag(), expected_imag, 1e-14);
}

// ============================================================================
// Symmetry Tests
// ============================================================================

TEST_F(CosPiTest, EvenSymmetry) {
  // cos_pi(-x) = cos_pi(x)
  std::vector<double> test_values = {0.1, 0.25, 0.5, 0.75, 1.5, 2.3};
  for (double x : test_values) {
    EXPECT_NEAR(cos_pi(-x), cos_pi(x), 1e-14)
        << "Even symmetry failed for x = " << x;
  }
}

TEST_F(CosPiTest, Periodicity) {
  // cos_pi(x + 2) = cos_pi(x)
  std::vector<double> test_values = {0.1, 0.25, 0.5, 0.75, 1.5};
  for (double x : test_values) {
    EXPECT_NEAR(cos_pi(x + 2.0), cos_pi(x), 1e-14)
        << "Periodicity failed for x = " << x;
    EXPECT_NEAR(cos_pi(x - 2.0), cos_pi(x), 1e-14)
        << "Periodicity failed for x = " << x;
  }
}

// ============================================================================
// Relationship with sin_pi Tests
// ============================================================================

TEST_F(CosPiTest, PythagoreanIdentity) {
  // sin_pi(x)^2 + cos_pi(x)^2 = 1
  std::vector<double> test_values = {0.0, 0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0};
  for (double x : test_values) {
    double s = sin_pi(x);
    double c = cos_pi(x);
    EXPECT_NEAR(s * s + c * c, 1.0, 1e-14)
        << "Pythagorean identity failed for x = " << x;
  }
}

TEST_F(CosPiTest, PhaseShift) {
  // cos_pi(x) = sin_pi(x + 0.5)
  std::vector<double> test_values = {0.0, 0.1, 0.25, 0.33, 0.5, 0.67, 0.75};
  for (double x : test_values) {
    EXPECT_NEAR(cos_pi(x), sin_pi(x + 0.5), 1e-14)
        << "Phase shift failed for x = " << x;
  }
}
