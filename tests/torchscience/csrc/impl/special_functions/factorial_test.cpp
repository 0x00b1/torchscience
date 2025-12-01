#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/factorial.h"
#include "impl/special_functions/gamma.h"

using namespace torchscience::impl::special_functions;

class FactorialTest : public ::testing::Test {};

// ============================================================================
// Float Tests - Integer Arguments
// ============================================================================

TEST_F(FactorialTest, Float_ZeroFactorial) {
  EXPECT_FLOAT_EQ(factorial(0.0f), 1.0f);
}

TEST_F(FactorialTest, Float_OneFactorial) {
  EXPECT_FLOAT_EQ(factorial(1.0f), 1.0f);
}

TEST_F(FactorialTest, Float_SmallIntegers) {
  EXPECT_FLOAT_EQ(factorial(2.0f), 2.0f);
  EXPECT_FLOAT_EQ(factorial(3.0f), 6.0f);
  EXPECT_FLOAT_EQ(factorial(4.0f), 24.0f);
  EXPECT_FLOAT_EQ(factorial(5.0f), 120.0f);
  EXPECT_FLOAT_EQ(factorial(6.0f), 720.0f);
  EXPECT_FLOAT_EQ(factorial(7.0f), 5040.0f);
  EXPECT_FLOAT_EQ(factorial(8.0f), 40320.0f);
  EXPECT_FLOAT_EQ(factorial(9.0f), 362880.0f);
  EXPECT_FLOAT_EQ(factorial(10.0f), 3628800.0f);
}

TEST_F(FactorialTest, Float_LUTBoundary) {
  // 34! is the last value in float LUT
  float factorial_34 = factorial(34.0f);
  EXPECT_FALSE(std::isinf(factorial_34));
  EXPECT_GT(factorial_34, 0.0f);

  // 35! should overflow to infinity
  float factorial_35 = factorial(35.0f);
  EXPECT_TRUE(std::isinf(factorial_35));
}

TEST_F(FactorialTest, Float_NegativeIntegers) {
  // Negative integers are poles
  EXPECT_TRUE(std::isinf(factorial(-1.0f)));
  EXPECT_TRUE(std::isinf(factorial(-2.0f)));
  EXPECT_TRUE(std::isinf(factorial(-5.0f)));
}

TEST_F(FactorialTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(factorial(nan_val)));
}

// ============================================================================
// Float Tests - Non-Integer Arguments
// ============================================================================

TEST_F(FactorialTest, Float_HalfInteger) {
  // (1/2)! = Gamma(3/2) = sqrt(pi)/2
  float expected = std::sqrt(static_cast<float>(M_PI)) / 2.0f;
  EXPECT_NEAR(factorial(0.5f), expected, 1e-5f);
}

TEST_F(FactorialTest, Float_NonIntegerPositive) {
  // z! = Gamma(z + 1)
  float z = 2.5f;
  float expected = gamma(z + 1.0f);
  EXPECT_NEAR(factorial(z), expected, 1e-5f);
}

// ============================================================================
// Double Tests - Integer Arguments
// ============================================================================

TEST_F(FactorialTest, Double_ZeroFactorial) {
  EXPECT_DOUBLE_EQ(factorial(0.0), 1.0);
}

TEST_F(FactorialTest, Double_OneFactorial) {
  EXPECT_DOUBLE_EQ(factorial(1.0), 1.0);
}

TEST_F(FactorialTest, Double_SmallIntegers) {
  EXPECT_DOUBLE_EQ(factorial(2.0), 2.0);
  EXPECT_DOUBLE_EQ(factorial(3.0), 6.0);
  EXPECT_DOUBLE_EQ(factorial(4.0), 24.0);
  EXPECT_DOUBLE_EQ(factorial(5.0), 120.0);
  EXPECT_DOUBLE_EQ(factorial(10.0), 3628800.0);
}

TEST_F(FactorialTest, Double_MediumIntegers) {
  // Check some values from the LUT
  EXPECT_DOUBLE_EQ(factorial(20.0), 2432902008176640000.0);
}

TEST_F(FactorialTest, Double_LUTBoundary) {
  // 170! is the last value in double LUT
  double factorial_170 = factorial(170.0);
  EXPECT_FALSE(std::isinf(factorial_170));
  EXPECT_GT(factorial_170, 0.0);

  // 171! should overflow to infinity
  double factorial_171 = factorial(171.0);
  EXPECT_TRUE(std::isinf(factorial_171));
}

TEST_F(FactorialTest, Double_NegativeIntegers) {
  EXPECT_TRUE(std::isinf(factorial(-1.0)));
  EXPECT_TRUE(std::isinf(factorial(-2.0)));
  EXPECT_TRUE(std::isinf(factorial(-10.0)));
}

TEST_F(FactorialTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(factorial(nan_val)));
}

// ============================================================================
// Double Tests - Non-Integer Arguments
// ============================================================================

TEST_F(FactorialTest, Double_HalfInteger) {
  // (1/2)! = Gamma(3/2) = sqrt(pi)/2
  double expected = std::sqrt(M_PI) / 2.0;
  EXPECT_NEAR(factorial(0.5), expected, 1e-14);
}

TEST_F(FactorialTest, Double_NegativeHalfInteger) {
  // (-1/2)! = Gamma(1/2) = sqrt(pi)
  double expected = std::sqrt(M_PI);
  EXPECT_NEAR(factorial(-0.5), expected, 1e-14);
}

TEST_F(FactorialTest, Double_NonIntegerPositive) {
  double z = 2.5;
  double expected = gamma(z + 1.0);
  EXPECT_NEAR(factorial(z), expected, 1e-14);
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(FactorialTest, ComplexFloat_RealIntegers) {
  c10::complex<float> z(5.0f, 0.0f);
  auto result = factorial(z);
  EXPECT_NEAR(result.real(), 120.0f, 1e-4f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(FactorialTest, ComplexFloat_ZeroReal) {
  c10::complex<float> z(0.0f, 0.0f);
  auto result = factorial(z);
  EXPECT_NEAR(result.real(), 1.0f, 1e-6f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(FactorialTest, ComplexFloat_NonZeroImaginary) {
  c10::complex<float> z(2.0f, 1.0f);
  auto result = factorial(z);
  // z! = Gamma(z + 1)
  auto expected = gamma(z + c10::complex<float>(1.0f, 0.0f));
  EXPECT_NEAR(result.real(), expected.real(), 1e-4f);
  EXPECT_NEAR(result.imag(), expected.imag(), 1e-4f);
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(FactorialTest, ComplexDouble_RealIntegers) {
  c10::complex<double> z(5.0, 0.0);
  auto result = factorial(z);
  EXPECT_NEAR(result.real(), 120.0, 1e-12);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(FactorialTest, ComplexDouble_ZeroReal) {
  c10::complex<double> z(0.0, 0.0);
  auto result = factorial(z);
  EXPECT_NEAR(result.real(), 1.0, 1e-14);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

TEST_F(FactorialTest, ComplexDouble_NonZeroImaginary) {
  c10::complex<double> z(2.0, 1.0);
  auto result = factorial(z);
  auto expected = gamma(z + c10::complex<double>(1.0, 0.0));
  EXPECT_NEAR(result.real(), expected.real(), 1e-12);
  EXPECT_NEAR(result.imag(), expected.imag(), 1e-12);
}

TEST_F(FactorialTest, ComplexDouble_NegativeIntegerPoles) {
  // At negative integers, factorial should return infinity
  c10::complex<double> z(-1.0, 0.0);
  auto result = factorial(z);
  EXPECT_TRUE(std::isinf(result.real()));
}

// ============================================================================
// Recurrence Relation Tests
// ============================================================================

TEST_F(FactorialTest, Double_RecurrenceRelation) {
  // n! = n * (n-1)!
  for (int n = 1; n <= 20; ++n) {
    double lhs = factorial(static_cast<double>(n));
    double rhs = n * factorial(static_cast<double>(n - 1));
    EXPECT_NEAR(lhs, rhs, lhs * 1e-14) << "Failed for n = " << n;
  }
}

TEST_F(FactorialTest, Float_RecurrenceRelation) {
  for (int n = 1; n <= 10; ++n) {
    float lhs = factorial(static_cast<float>(n));
    float rhs = n * factorial(static_cast<float>(n - 1));
    EXPECT_NEAR(lhs, rhs, lhs * 1e-6f) << "Failed for n = " << n;
  }
}

// ============================================================================
// Gamma Function Relationship Tests
// ============================================================================

TEST_F(FactorialTest, Double_GammaRelation) {
  // z! = Gamma(z + 1) for non-negative integers
  for (int n = 0; n <= 10; ++n) {
    double z = static_cast<double>(n);
    double factorial_result = factorial(z);
    double gamma_result = gamma(z + 1.0);
    EXPECT_NEAR(factorial_result, gamma_result, gamma_result * 1e-14)
        << "Failed for n = " << n;
  }
}

TEST_F(FactorialTest, Double_GammaRelationNonInteger) {
  std::vector<double> test_values = {0.5, 1.5, 2.5, 3.7, 4.2};
  for (double z : test_values) {
    double factorial_result = factorial(z);
    double gamma_result = gamma(z + 1.0);
    EXPECT_NEAR(factorial_result, gamma_result, std::abs(gamma_result) * 1e-14)
        << "Failed for z = " << z;
  }
}
