#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/sign_gamma.h"

using namespace torchscience::impl::special_functions;

class SignGammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(SignGammaTest, Float_PositiveValues) {
  // Gamma(x) > 0 for all x > 0
  EXPECT_EQ(sign_gamma(0.5f), 1);
  EXPECT_EQ(sign_gamma(1.0f), 1);
  EXPECT_EQ(sign_gamma(1.5f), 1);
  EXPECT_EQ(sign_gamma(2.0f), 1);
  EXPECT_EQ(sign_gamma(3.0f), 1);
  EXPECT_EQ(sign_gamma(10.0f), 1);
  EXPECT_EQ(sign_gamma(100.0f), 1);
}

TEST_F(SignGammaTest, Float_NegativeNonIntegers) {
  // Between -1 and 0: Gamma < 0
  EXPECT_EQ(sign_gamma(-0.5f), -1);
  EXPECT_EQ(sign_gamma(-0.1f), -1);
  EXPECT_EQ(sign_gamma(-0.9f), -1);

  // Between -2 and -1: Gamma > 0
  EXPECT_EQ(sign_gamma(-1.5f), 1);
  EXPECT_EQ(sign_gamma(-1.1f), 1);
  EXPECT_EQ(sign_gamma(-1.9f), 1);

  // Between -3 and -2: Gamma < 0
  EXPECT_EQ(sign_gamma(-2.5f), -1);
  EXPECT_EQ(sign_gamma(-2.1f), -1);
  EXPECT_EQ(sign_gamma(-2.9f), -1);

  // Between -4 and -3: Gamma > 0
  EXPECT_EQ(sign_gamma(-3.5f), 1);
  EXPECT_EQ(sign_gamma(-3.1f), 1);
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(SignGammaTest, Double_PositiveValues) {
  EXPECT_EQ(sign_gamma(0.5), 1);
  EXPECT_EQ(sign_gamma(1.0), 1);
  EXPECT_EQ(sign_gamma(1.5), 1);
  EXPECT_EQ(sign_gamma(2.0), 1);
  EXPECT_EQ(sign_gamma(3.0), 1);
  EXPECT_EQ(sign_gamma(10.0), 1);
  EXPECT_EQ(sign_gamma(100.0), 1);
}

TEST_F(SignGammaTest, Double_NegativeNonIntegers) {
  // Between -1 and 0: Gamma < 0
  EXPECT_EQ(sign_gamma(-0.5), -1);
  EXPECT_EQ(sign_gamma(-0.1), -1);
  EXPECT_EQ(sign_gamma(-0.9), -1);

  // Between -2 and -1: Gamma > 0
  EXPECT_EQ(sign_gamma(-1.5), 1);
  EXPECT_EQ(sign_gamma(-1.1), 1);
  EXPECT_EQ(sign_gamma(-1.9), 1);

  // Between -3 and -2: Gamma < 0
  EXPECT_EQ(sign_gamma(-2.5), -1);
  EXPECT_EQ(sign_gamma(-2.1), -1);
  EXPECT_EQ(sign_gamma(-2.9), -1);

  // Between -4 and -3: Gamma > 0
  EXPECT_EQ(sign_gamma(-3.5), 1);
  EXPECT_EQ(sign_gamma(-3.1), 1);

  // Between -5 and -4: Gamma < 0
  EXPECT_EQ(sign_gamma(-4.5), -1);
}

// ============================================================================
// Pattern Verification Tests
// ============================================================================

TEST_F(SignGammaTest, Double_AlternatingPattern) {
  // Verify the alternating pattern for negative non-integers
  for (int k = 0; k < 10; ++k) {
    double x = -static_cast<double>(k) - 0.5;
    int expected_sign = (k % 2 == 0) ? -1 : 1;
    EXPECT_EQ(sign_gamma(x), expected_sign)
        << "Failed for x = " << x;
  }
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(SignGammaTest, Double_ConsistencyWithGamma) {
  // Verify that sign_gamma gives the correct sign of the actual gamma function
  std::vector<double> test_values = {
      0.5, 1.5, 2.5,          // Positive values
      -0.5, -0.1, -0.9,       // (-1, 0)
      -1.5, -1.1, -1.9,       // (-2, -1)
      -2.5, -2.1, -2.9,       // (-3, -2)
      -3.5                    // (-4, -3)
  };

  for (double x : test_values) {
    double gamma_val = std::tgamma(x);
    int expected = (gamma_val > 0) ? 1 : -1;
    EXPECT_EQ(sign_gamma(x), expected)
        << "Failed for x = " << x << ", gamma = " << gamma_val;
  }
}

// ============================================================================
// signed_exp_lgamma Tests
// ============================================================================

TEST_F(SignGammaTest, Double_SignedExpLgamma_PositiveValues) {
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0};
  for (double x : test_values) {
    double result = signed_exp_lgamma(x);
    double expected = std::tgamma(x);
    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-7)
        << "Failed for x = " << x;
  }
}

TEST_F(SignGammaTest, Double_SignedExpLgamma_NegativeValues) {
  std::vector<double> test_values = {-0.5, -1.5, -2.5, -3.5};
  for (double x : test_values) {
    double result = signed_exp_lgamma(x);
    double expected = std::tgamma(x);
    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-6)
        << "Failed for x = " << x;
  }
}

// ============================================================================
// gamma_ratio Tests
// ============================================================================

TEST_F(SignGammaTest, Double_GammaRatio_PositiveValues) {
  // Gamma(a) / Gamma(b)
  double a = 5.0;
  double b = 3.0;
  double result = gamma_ratio(a, b);
  double expected = std::tgamma(a) / std::tgamma(b);
  EXPECT_NEAR(result, expected, std::abs(expected) * 1e-10);
}

TEST_F(SignGammaTest, Double_GammaRatio_MixedValues) {
  double a = 2.5;
  double b = -0.5;
  double result = gamma_ratio(a, b);
  double expected = std::tgamma(a) / std::tgamma(b);
  EXPECT_NEAR(result, expected, std::abs(expected) * 1e-6);
}

// ============================================================================
// gamma_ratio_4 Tests
// ============================================================================

TEST_F(SignGammaTest, Double_GammaRatio4_PositiveValues) {
  // (Gamma(a1) * Gamma(a2)) / (Gamma(b1) * Gamma(b2))
  double a1 = 3.0, a2 = 4.0, b1 = 2.0, b2 = 5.0;
  double result = gamma_ratio_4(a1, a2, b1, b2);
  double expected = (std::tgamma(a1) * std::tgamma(a2)) /
                    (std::tgamma(b1) * std::tgamma(b2));
  EXPECT_NEAR(result, expected, std::abs(expected) * 1e-10);
}

TEST_F(SignGammaTest, Double_GammaRatio4_MixedValues) {
  double a1 = 2.5, a2 = 1.5, b1 = 0.5, b2 = 3.5;
  double result = gamma_ratio_4(a1, a2, b1, b2);
  double expected = (std::tgamma(a1) * std::tgamma(a2)) /
                    (std::tgamma(b1) * std::tgamma(b2));
  EXPECT_NEAR(result, expected, std::abs(expected) * 1e-6);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SignGammaTest, Float_SmallPositive) {
  EXPECT_EQ(sign_gamma(0.001f), 1);
  EXPECT_EQ(sign_gamma(1e-6f), 1);
}

TEST_F(SignGammaTest, Double_SmallPositive) {
  EXPECT_EQ(sign_gamma(0.001), 1);
  EXPECT_EQ(sign_gamma(1e-10), 1);
}

TEST_F(SignGammaTest, Double_LargeNegative) {
  // Very negative values
  EXPECT_EQ(sign_gamma(-10.5), -1);  // (-11, -10), k=10 (even), sign = -1
  EXPECT_EQ(sign_gamma(-11.5), 1);   // (-12, -11), k=11 (odd), sign = 1
}
