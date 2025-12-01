#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/incomplete_beta.h"

using namespace torchscience::impl::special_functions;

class IncompleteBetaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// log_beta Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_LogBeta_PositiveValues) {
  // log(B(a, b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
  double a = 2.0, b = 3.0;
  double result = log_beta(a, b);
  double expected = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  EXPECT_NEAR(result, expected, 1e-14);
}

TEST_F(IncompleteBetaTest, Double_LogBeta_Symmetry) {
  // B(a, b) = B(b, a), so log(B(a, b)) = log(B(b, a))
  double a = 2.5, b = 3.5;
  double result1 = log_beta(a, b);
  double result2 = log_beta(b, a);
  EXPECT_NEAR(result1, result2, 1e-14);
}

TEST_F(IncompleteBetaTest, Double_LogBeta_SmallValues) {
  double a = 0.5, b = 0.5;
  double result = log_beta(a, b);
  // B(0.5, 0.5) = pi, so log(B(0.5, 0.5)) = log(pi)
  EXPECT_NEAR(result, std::log(kPi), 1e-14);
}

TEST_F(IncompleteBetaTest, Float_LogBeta_PositiveValues) {
  float a = 2.0f, b = 3.0f;
  float result = log_beta(a, b);
  float expected = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  EXPECT_NEAR(result, expected, 1e-5f);
}

// ============================================================================
// incomplete_beta Tests - Boundary Values
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_AtZero) {
  // I_0(a, b) = 0
  double a = 2.0, b = 3.0;
  double result = incomplete_beta(0.0, a, b);
  EXPECT_NEAR(result, 0.0, 1e-14);
}

TEST_F(IncompleteBetaTest, Double_IBeta_AtOne) {
  // I_1(a, b) = 1
  double a = 2.0, b = 3.0;
  double result = incomplete_beta(1.0, a, b);
  EXPECT_NEAR(result, 1.0, 1e-14);
}

// ============================================================================
// Special Cases
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_UniformDistribution) {
  // I_z(1, 1) = z (CDF of uniform distribution)
  std::vector<double> z_values = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (double z : z_values) {
    double result = incomplete_beta(z, 1.0, 1.0);
    EXPECT_NEAR(result, z, 1e-12) << "Failed for z = " << z;
  }
}

TEST_F(IncompleteBetaTest, Double_IBeta_AIsOne) {
  // I_z(1, b) = 1 - (1-z)^b
  double b = 3.0;
  std::vector<double> z_values = {0.1, 0.25, 0.5, 0.75, 0.9};
  for (double z : z_values) {
    double result = incomplete_beta(z, 1.0, b);
    double expected = 1.0 - std::pow(1.0 - z, b);
    EXPECT_NEAR(result, expected, 1e-10) << "Failed for z = " << z;
  }
}

TEST_F(IncompleteBetaTest, Double_IBeta_BIsOne) {
  // I_z(a, 1) = z^a
  double a = 3.0;
  std::vector<double> z_values = {0.1, 0.25, 0.5, 0.75, 0.9};
  for (double z : z_values) {
    double result = incomplete_beta(z, a, 1.0);
    double expected = std::pow(z, a);
    EXPECT_NEAR(result, expected, 1e-10) << "Failed for z = " << z;
  }
}

// ============================================================================
// Symmetry Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_Symmetry) {
  // I_z(a, b) + I_{1-z}(b, a) = 1
  double a = 2.5, b = 3.5;
  std::vector<double> z_values = {0.1, 0.25, 0.5, 0.75, 0.9};
  for (double z : z_values) {
    double i_z = incomplete_beta(z, a, b);
    double i_1_minus_z = incomplete_beta(1.0 - z, b, a);
    EXPECT_NEAR(i_z + i_1_minus_z, 1.0, 1e-10) << "Failed for z = " << z;
  }
}

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Float_IBeta_AtZero) {
  float a = 2.0f, b = 3.0f;
  float result = incomplete_beta(0.0f, a, b);
  EXPECT_NEAR(result, 0.0f, 1e-6f);
}

TEST_F(IncompleteBetaTest, Float_IBeta_AtOne) {
  float a = 2.0f, b = 3.0f;
  float result = incomplete_beta(1.0f, a, b);
  EXPECT_NEAR(result, 1.0f, 1e-6f);
}

TEST_F(IncompleteBetaTest, Float_IBeta_Middle) {
  float a = 2.0f, b = 3.0f;
  float result = incomplete_beta(0.5f, a, b);
  EXPECT_GT(result, 0.0f);
  EXPECT_LT(result, 1.0f);
  EXPECT_FALSE(std::isnan(result));
}

// ============================================================================
// Range Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_ResultInRange) {
  // Result should always be in [0, 1]
  std::vector<double> a_values = {0.5, 1.0, 2.0, 5.0};
  std::vector<double> b_values = {0.5, 1.0, 2.0, 5.0};
  std::vector<double> z_values = {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0};

  for (double a : a_values) {
    for (double b : b_values) {
      for (double z : z_values) {
        double result = incomplete_beta(z, a, b);
        EXPECT_GE(result, 0.0 - 1e-10)
            << "Failed for z=" << z << ", a=" << a << ", b=" << b;
        EXPECT_LE(result, 1.0 + 1e-10)
            << "Failed for z=" << z << ", a=" << a << ", b=" << b;
      }
    }
  }
}

// ============================================================================
// Monotonicity Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_Monotonic) {
  // I_z(a, b) should be monotonically increasing in z
  double a = 2.0, b = 3.0;
  std::vector<double> z_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  double prev_result = -1.0;
  for (double z : z_values) {
    double result = incomplete_beta(z, a, b);
    EXPECT_GE(result, prev_result - 1e-10)
        << "Not monotonic at z = " << z;
    prev_result = result;
  }
}

// ============================================================================
// Small Parameter Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_SmallA) {
  double a = 0.1, b = 2.0, z = 0.5;
  double result = incomplete_beta(z, a, b);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
  EXPECT_GT(result, 0.0);
  EXPECT_LT(result, 1.0);
}

TEST_F(IncompleteBetaTest, Double_IBeta_SmallB) {
  double a = 2.0, b = 0.1, z = 0.5;
  double result = incomplete_beta(z, a, b);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
  EXPECT_GT(result, 0.0);
  EXPECT_LT(result, 1.0);
}

TEST_F(IncompleteBetaTest, Double_IBeta_BothSmall) {
  double a = 0.1, b = 0.1, z = 0.5;
  double result = incomplete_beta(z, a, b);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

// ============================================================================
// Large Parameter Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_LargeParameters) {
  double a = 100.0, b = 100.0, z = 0.5;
  double result = incomplete_beta(z, a, b);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
  // For symmetric large parameters at z=0.5, result should be close to 0.5
  EXPECT_NEAR(result, 0.5, 0.1);
}

TEST_F(IncompleteBetaTest, Double_IBeta_AsymmetricLarge) {
  double a = 50.0, b = 100.0, z = 0.3;
  double result = incomplete_beta(z, a, b);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
  EXPECT_GT(result, 0.0);
  EXPECT_LT(result, 1.0);
}

// ============================================================================
// Invalid Input Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Double_IBeta_OutOfRangeZ) {
  double a = 2.0, b = 3.0;
  // z < 0 returns 0 (clamped to lower bound)
  double result_neg = incomplete_beta(-0.1, a, b);
  EXPECT_NEAR(result_neg, 0.0, 1e-14);

  // z > 1 returns 1 (clamped to upper bound)
  double result_large = incomplete_beta(1.1, a, b);
  EXPECT_NEAR(result_large, 1.0, 1e-14);
}

TEST_F(IncompleteBetaTest, Double_IBeta_InvalidA) {
  // a <= 0 should return NaN
  double result = incomplete_beta(0.5, 0.0, 2.0);
  EXPECT_TRUE(std::isnan(result));

  double result_neg = incomplete_beta(0.5, -1.0, 2.0);
  EXPECT_TRUE(std::isnan(result_neg));
}

TEST_F(IncompleteBetaTest, Double_IBeta_InvalidB) {
  // b <= 0 should return NaN
  double result = incomplete_beta(0.5, 2.0, 0.0);
  EXPECT_TRUE(std::isnan(result));

  double result_neg = incomplete_beta(0.5, 2.0, -1.0);
  EXPECT_TRUE(std::isnan(result_neg));
}

// ============================================================================
// Complex Tests
// ============================================================================

TEST_F(IncompleteBetaTest, ComplexDouble_LogBeta) {
  c10::complex<double> a(2.0, 0.0);
  c10::complex<double> b(3.0, 0.0);
  auto result = log_beta(a, b);
  double expected = std::lgamma(2.0) + std::lgamma(3.0) - std::lgamma(5.0);
  EXPECT_NEAR(result.real(), expected, 1e-12);
  EXPECT_NEAR(result.imag(), 0.0, 1e-14);
}

// ============================================================================
// Half/BFloat16 Tests
// ============================================================================

TEST_F(IncompleteBetaTest, Float_LogBeta_Consistency) {
  // Float result should be consistent with double
  float a_f = 2.0f, b_f = 3.0f;
  double a_d = 2.0, b_d = 3.0;

  float result_f = log_beta(a_f, b_f);
  double result_d = log_beta(a_d, b_d);

  EXPECT_NEAR(result_f, result_d, 1e-5);
}
