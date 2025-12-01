#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/is_nonpositive_integer.h"

using namespace torchscience::impl::special_functions;

class IsNonpositiveIntegerTest : public ::testing::Test {};

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_Zero) {
  c10::complex<float> z(0.0f, 0.0f);
  EXPECT_TRUE(is_nonpositive_integer(z));
}

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_NegativeIntegers) {
  for (int n = -10; n <= 0; ++n) {
    c10::complex<float> z(static_cast<float>(n), 0.0f);
    EXPECT_TRUE(is_nonpositive_integer(z)) << "Failed for n = " << n;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_PositiveIntegers) {
  for (int n = 1; n <= 10; ++n) {
    c10::complex<float> z(static_cast<float>(n), 0.0f);
    EXPECT_FALSE(is_nonpositive_integer(z)) << "Failed for n = " << n;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_NonIntegers) {
  std::vector<float> values = {0.5f, -0.5f, 0.1f, -0.1f, -1.5f, -2.5f, 0.999f, -0.001f};
  for (float x : values) {
    c10::complex<float> z(x, 0.0f);
    EXPECT_FALSE(is_nonpositive_integer(z)) << "Failed for x = " << x;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_NonZeroImaginary) {
  // Any non-zero imaginary part means it's not a non-positive integer
  c10::complex<float> z1(0.0f, 0.1f);
  EXPECT_FALSE(is_nonpositive_integer(z1));

  c10::complex<float> z2(-1.0f, 0.1f);
  EXPECT_FALSE(is_nonpositive_integer(z2));

  c10::complex<float> z3(-1.0f, 1.0f);
  EXPECT_FALSE(is_nonpositive_integer(z3));
}

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_NearIntegerWithTinyImaginary) {
  // Very small imaginary part should still be detected
  float tol = kPoleDetectionToleranceFloat;
  c10::complex<float> z1(-1.0f, tol * 0.5f);
  EXPECT_TRUE(is_nonpositive_integer(z1));

  c10::complex<float> z2(-1.0f, tol * 2.0f);
  EXPECT_FALSE(is_nonpositive_integer(z2));
}

// ============================================================================
// Complex Double Tests
// ============================================================================

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_Zero) {
  c10::complex<double> z(0.0, 0.0);
  EXPECT_TRUE(is_nonpositive_integer(z));
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_NegativeIntegers) {
  for (int n = -10; n <= 0; ++n) {
    c10::complex<double> z(static_cast<double>(n), 0.0);
    EXPECT_TRUE(is_nonpositive_integer(z)) << "Failed for n = " << n;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_PositiveIntegers) {
  for (int n = 1; n <= 10; ++n) {
    c10::complex<double> z(static_cast<double>(n), 0.0);
    EXPECT_FALSE(is_nonpositive_integer(z)) << "Failed for n = " << n;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_NonIntegers) {
  std::vector<double> values = {0.5, -0.5, 0.1, -0.1, -1.5, -2.5, 0.999, -0.001};
  for (double x : values) {
    c10::complex<double> z(x, 0.0);
    EXPECT_FALSE(is_nonpositive_integer(z)) << "Failed for x = " << x;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_NonZeroImaginary) {
  c10::complex<double> z1(0.0, 0.1);
  EXPECT_FALSE(is_nonpositive_integer(z1));

  c10::complex<double> z2(-1.0, 0.1);
  EXPECT_FALSE(is_nonpositive_integer(z2));
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_NearIntegerWithTinyImaginary) {
  double tol = kPoleDetectionToleranceDouble;
  c10::complex<double> z1(-1.0, tol * 0.5);
  EXPECT_TRUE(is_nonpositive_integer(z1));

  c10::complex<double> z2(-1.0, tol * 2.0);
  EXPECT_FALSE(is_nonpositive_integer(z2));
}

// ============================================================================
// Large Value Tests
// ============================================================================

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_LargeNegativeIntegers) {
  // Test large negative integers where relative tolerance matters
  c10::complex<double> z1(-100.0, 0.0);
  EXPECT_TRUE(is_nonpositive_integer(z1));

  c10::complex<double> z2(-1000.0, 0.0);
  EXPECT_TRUE(is_nonpositive_integer(z2));
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_NearLargeIntegers) {
  // Test values very close to large negative integers
  double tol = kPoleDetectionToleranceDouble;

  // Should be detected as -100
  c10::complex<double> z1(-100.0 + tol * 50.0, 0.0);
  EXPECT_TRUE(is_nonpositive_integer(z1));

  // Should NOT be detected as integer (too far)
  c10::complex<double> z2(-100.5, 0.0);
  EXPECT_FALSE(is_nonpositive_integer(z2));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_HalfIntegers) {
  // Half-integers should not be detected
  std::vector<double> half_ints = {-0.5, -1.5, -2.5, -3.5, -10.5};
  for (double x : half_ints) {
    c10::complex<double> z(x, 0.0);
    EXPECT_FALSE(is_nonpositive_integer(z)) << "Failed for x = " << x;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexFloat_HalfIntegers) {
  std::vector<float> half_ints = {-0.5f, -1.5f, -2.5f, -3.5f, -10.5f};
  for (float x : half_ints) {
    c10::complex<float> z(x, 0.0f);
    EXPECT_FALSE(is_nonpositive_integer(z)) << "Failed for x = " << x;
  }
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_VerySmallPositive) {
  // Very small positive numbers should not be detected as zero
  double small = 1e-10;
  c10::complex<double> z(small, 0.0);
  EXPECT_FALSE(is_nonpositive_integer(z));
}

TEST_F(IsNonpositiveIntegerTest, ComplexDouble_NearZero) {
  // Values very close to zero should be detected
  double tol = kPoleDetectionToleranceDouble;
  c10::complex<double> z1(tol * 0.5, 0.0);
  EXPECT_TRUE(is_nonpositive_integer(z1));

  c10::complex<double> z2(-tol * 0.5, 0.0);
  EXPECT_TRUE(is_nonpositive_integer(z2));
}
