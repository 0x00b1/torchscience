#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/chebyshev_polynomial_t.h"

using namespace torchscience::impl::special_functions;

class ChebyshevPolynomialTTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Float Tests - Integer Degrees
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Float_T0) {
  // T_0(z) = 1
  std::vector<float> test_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  for (float z : test_values) {
    EXPECT_FLOAT_EQ(chebyshev_polynomial_t(0.0f, z), 1.0f)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Float_T1) {
  // T_1(z) = z
  std::vector<float> test_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  for (float z : test_values) {
    EXPECT_FLOAT_EQ(chebyshev_polynomial_t(1.0f, z), z)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Float_T2) {
  // T_2(z) = 2z^2 - 1
  std::vector<float> test_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  for (float z : test_values) {
    float expected = 2.0f * z * z - 1.0f;
    EXPECT_NEAR(chebyshev_polynomial_t(2.0f, z), expected, 1e-6f)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Float_T3) {
  // T_3(z) = 4z^3 - 3z
  std::vector<float> test_values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  for (float z : test_values) {
    float expected = 4.0f * z * z * z - 3.0f * z;
    EXPECT_NEAR(chebyshev_polynomial_t(3.0f, z), expected, 1e-5f)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Float_BoundaryValues) {
  // T_n(1) = 1 for all n
  for (int n = 0; n <= 10; ++n) {
    EXPECT_NEAR(chebyshev_polynomial_t(static_cast<float>(n), 1.0f), 1.0f, 1e-6f)
        << "Failed for n = " << n;
  }

  // T_n(-1) = (-1)^n for all n
  for (int n = 0; n <= 10; ++n) {
    float expected = (n % 2 == 0) ? 1.0f : -1.0f;
    EXPECT_NEAR(chebyshev_polynomial_t(static_cast<float>(n), -1.0f), expected, 1e-6f)
        << "Failed for n = " << n;
  }
}

TEST_F(ChebyshevPolynomialTTest, Float_NegativeIndex) {
  // T_{-n}(z) = T_n(z)
  std::vector<float> test_values = {-0.5f, 0.0f, 0.5f};
  for (float z : test_values) {
    for (int n = 1; n <= 5; ++n) {
      float pos = chebyshev_polynomial_t(static_cast<float>(n), z);
      float neg = chebyshev_polynomial_t(static_cast<float>(-n), z);
      EXPECT_NEAR(pos, neg, 1e-5f)
          << "Failed for n = " << n << ", z = " << z;
    }
  }
}

// ============================================================================
// Double Tests - Integer Degrees
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_T0) {
  std::vector<double> test_values = {-1.0, -0.5, 0.0, 0.5, 1.0};
  for (double z : test_values) {
    EXPECT_DOUBLE_EQ(chebyshev_polynomial_t(0.0, z), 1.0)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Double_T1) {
  std::vector<double> test_values = {-1.0, -0.5, 0.0, 0.5, 1.0};
  for (double z : test_values) {
    EXPECT_DOUBLE_EQ(chebyshev_polynomial_t(1.0, z), z)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Double_T2) {
  std::vector<double> test_values = {-1.0, -0.5, 0.0, 0.5, 1.0};
  for (double z : test_values) {
    double expected = 2.0 * z * z - 1.0;
    EXPECT_NEAR(chebyshev_polynomial_t(2.0, z), expected, 1e-14)
        << "Failed for z = " << z;
  }
}

TEST_F(ChebyshevPolynomialTTest, Double_LargerDegrees) {
  // Test higher degrees with explicit formulas or recurrence check
  double z = 0.5;

  // T_4(z) = 8z^4 - 8z^2 + 1
  double z2 = z * z;
  double expected_t4 = 8.0 * z2 * z2 - 8.0 * z2 + 1.0;
  EXPECT_NEAR(chebyshev_polynomial_t(4.0, z), expected_t4, 1e-14);

  // T_5(z) = 16z^5 - 20z^3 + 5z
  double expected_t5 = 16.0 * z2 * z2 * z - 20.0 * z2 * z + 5.0 * z;
  EXPECT_NEAR(chebyshev_polynomial_t(5.0, z), expected_t5, 1e-14);
}

// ============================================================================
// Analytic Continuation Tests (Non-Integer v)
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_NonIntegerV_InDomain) {
  // T_v(z) = cos(v * acos(z)) for |z| <= 1
  double v = 2.5;
  double z = 0.5;
  double expected = std::cos(v * std::acos(z));
  EXPECT_NEAR(chebyshev_polynomial_t(v, z), expected, 1e-14);
}

TEST_F(ChebyshevPolynomialTTest, Double_NonIntegerV_AtZero) {
  // T_v(0) = cos(v * pi/2)
  double v = 1.5;
  double expected = std::cos(v * kPi / 2.0);
  EXPECT_NEAR(chebyshev_polynomial_t(v, 0.0), expected, 1e-14);
}

// ============================================================================
// Hyperbolic Continuation Tests (|z| > 1)
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_IntegerDegree_OutsideDomain) {
  // For integer degrees, recurrence works for all real z
  double z = 2.0;
  double result = chebyshev_polynomial_t(3.0, z);
  // T_3(2) = 4*8 - 3*2 = 32 - 6 = 26
  EXPECT_NEAR(result, 26.0, 1e-12);
}

TEST_F(ChebyshevPolynomialTTest, Double_NonIntegerV_GreaterThanOne) {
  // T_v(z) = cosh(v * acosh(z)) for z > 1
  double v = 1.5;
  double z = 2.0;
  double expected = std::cosh(v * std::acosh(z));
  EXPECT_NEAR(chebyshev_polynomial_t(v, z), expected, 1e-12);
}

TEST_F(ChebyshevPolynomialTTest, Double_NonIntegerV_LessThanMinusOne) {
  // T_v(z) = cos(v*pi) * cosh(v * acosh(-z)) for z < -1
  double v = 1.5;
  double z = -2.0;
  double expected = std::cos(v * kPi) * std::cosh(v * std::acosh(-z));
  EXPECT_NEAR(chebyshev_polynomial_t(v, z), expected, 1e-12);
}

// ============================================================================
// Complex Tests
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, ComplexFloat_T0) {
  c10::complex<float> z(0.5f, 0.25f);
  auto result = chebyshev_polynomial_t(c10::complex<float>(0.0f, 0.0f), z);
  EXPECT_NEAR(result.real(), 1.0f, 1e-6f);
  EXPECT_NEAR(result.imag(), 0.0f, 1e-6f);
}

TEST_F(ChebyshevPolynomialTTest, ComplexDouble_T1) {
  c10::complex<double> z(0.5, 0.25);
  auto result = chebyshev_polynomial_t(c10::complex<double>(1.0, 0.0), z);
  EXPECT_NEAR(result.real(), 0.5, 1e-14);
  EXPECT_NEAR(result.imag(), 0.25, 1e-14);
}

TEST_F(ChebyshevPolynomialTTest, ComplexDouble_General) {
  // T_v(z) = cos(v * acos(z)) using complex arithmetic
  c10::complex<double> v(2.0, 0.0);
  c10::complex<double> z(0.5, 0.25);
  auto result = chebyshev_polynomial_t(v, z);
  EXPECT_FALSE(std::isnan(result.real()));
  EXPECT_FALSE(std::isnan(result.imag()));
}

// ============================================================================
// Recurrence Relation Tests
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_RecurrenceRelation) {
  // T_{n+1}(z) = 2z * T_n(z) - T_{n-1}(z)
  std::vector<double> z_values = {-0.8, -0.5, 0.0, 0.5, 0.8};
  for (double z : z_values) {
    for (int n = 1; n <= 8; ++n) {
      double t_n_plus_1 = chebyshev_polynomial_t(static_cast<double>(n + 1), z);
      double t_n = chebyshev_polynomial_t(static_cast<double>(n), z);
      double t_n_minus_1 = chebyshev_polynomial_t(static_cast<double>(n - 1), z);
      double expected = 2.0 * z * t_n - t_n_minus_1;
      EXPECT_NEAR(t_n_plus_1, expected, 1e-12)
          << "Failed for n = " << n << ", z = " << z;
    }
  }
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_Backward) {
  double v = 3.0;
  double z = 0.5;
  double grad = 1.0;

  auto [grad_v, grad_z] = chebyshev_polynomial_t_backward(grad, v, z);

  // Finite difference check for grad_z
  double eps = 1e-7;
  double numerical_grad_z = (chebyshev_polynomial_t(v, z + eps) -
                             chebyshev_polynomial_t(v, z - eps)) / (2.0 * eps);
  EXPECT_NEAR(grad_z, numerical_grad_z, 1e-5);

  // Finite difference check for grad_v
  double numerical_grad_v = (chebyshev_polynomial_t(v + eps, z) -
                             chebyshev_polynomial_t(v - eps, z)) / (2.0 * eps);
  EXPECT_NEAR(grad_v, numerical_grad_v, 1e-5);
}

TEST_F(ChebyshevPolynomialTTest, Double_Backward_NonIntegerV) {
  double v = 2.5;
  double z = 0.5;
  double grad = 1.0;

  auto [grad_v, grad_z] = chebyshev_polynomial_t_backward(grad, v, z);

  double eps = 1e-7;
  double numerical_grad_z = (chebyshev_polynomial_t(v, z + eps) -
                             chebyshev_polynomial_t(v, z - eps)) / (2.0 * eps);
  EXPECT_NEAR(grad_z, numerical_grad_z, 1e-5);

  double numerical_grad_v = (chebyshev_polynomial_t(v + eps, z) -
                             chebyshev_polynomial_t(v - eps, z)) / (2.0 * eps);
  EXPECT_NEAR(grad_v, numerical_grad_v, 1e-5);
}

// ============================================================================
// Special Values Tests
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_AtCosValues) {
  // T_n(cos(theta)) = cos(n*theta)
  std::vector<double> theta_values = {0.0, kPi / 6, kPi / 4, kPi / 3, kPi / 2};
  for (double theta : theta_values) {
    double z = std::cos(theta);
    for (int n = 0; n <= 5; ++n) {
      double expected = std::cos(n * theta);
      double result = chebyshev_polynomial_t(static_cast<double>(n), z);
      EXPECT_NEAR(result, expected, 1e-12)
          << "Failed for n = " << n << ", theta = " << theta;
    }
  }
}

// ============================================================================
// Orthogonality Tests
// ============================================================================

TEST_F(ChebyshevPolynomialTTest, Double_Orthogonality) {
  // Chebyshev polynomials are orthogonal with weight 1/sqrt(1-x^2)
  // We just verify that values are bounded by [-1, 1] for |z| <= 1
  for (int n = 0; n <= 10; ++n) {
    std::vector<double> z_values = {-0.9, -0.5, 0.0, 0.5, 0.9};
    for (double z : z_values) {
      double result = chebyshev_polynomial_t(static_cast<double>(n), z);
      EXPECT_GE(result, -1.0 - 1e-10) << "n = " << n << ", z = " << z;
      EXPECT_LE(result, 1.0 + 1e-10) << "n = " << n << ", z = " << z;
    }
  }
}
